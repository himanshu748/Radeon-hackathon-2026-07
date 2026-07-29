"""Run LTX-2.3 image-to-video on a Radeon instance whose RAM ceiling is smaller
than the model pair it ships with.

The stock template pairs a 43 GB diffusion checkpoint with a 23 GB text encoder
inside a container capped at 55 GB, so the two never coexist: the load trips the
cap and the platform restarts the instance mid-prompt. Running the shipped
workflow unmodified takes the whole box down.

The fix is to never hold both at once:

    phase 1  load the text encoder, encode the prompts, write the conditioning
             to disk, exit so the process gives its RAM back
    phase 2  load the diffusion checkpoint, read the conditioning back, sample,
             write frames and audio, exit

Peak resident set is then max(43, 23) rather than 43 + 23, which fits under the
cap with room to spare.

LTX-2.3 is an audio-video model: one sampling pass produces a video latent and
an audio latent that are concatenated, denoised together and separated again.
The node graph here mirrors the workflow that ships with the template, so the
semantics match what the ComfyUI editor would run.

Runs on the instance itself and imports ComfyUI as a library rather than
starting its server, because the server holds every model a graph touches for
the lifetime of the process, which is the thing that does not fit.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
import wave
from pathlib import Path

COMFY = os.environ.get("COMFY_ROOT", "/comfyui_workspace/ComfyUI")

#: Steps for the base pass, taken from the shipped workflow. The distilled LoRA
#: is trained for this schedule, so these are not free parameters.
BASE_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
#: The refine pass starts part-way down the schedule, hence the 0.85 head.
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
DISTILL_LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def _boot():
    """Import ComfyUI as a library rather than starting its server."""
    sys.path.insert(0, COMFY)
    os.chdir(COMFY)
    # comfy.cli_args parses sys.argv at import, so hide our own flags from it.
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        import nodes  # noqa: F401

        init = getattr(nodes, "init_extra_nodes", None) or getattr(nodes, "init_custom_nodes", None)
        if init:
            r = init()
            if hasattr(r, "__await__"):
                import asyncio

                asyncio.run(r)
        return nodes
    finally:
        sys.argv = argv


def _call(nodes, name: str, **kwargs):
    """Invoke a ComfyUI node the way its own executor does.

    Nodes carry their entry point in FUNCTION and it moves between versions
    (`generate` on this box is a shim onto a classmethod with a different
    signature, which silently shifts positional arguments). Going through
    FUNCTION with keywords keeps this working across ComfyUI upgrades.
    """
    cls = nodes.NODE_CLASS_MAPPINGS[name]
    out = getattr(cls(), cls.FUNCTION)(**kwargs)
    return getattr(out, "result", out)


def _peak_gb() -> float:
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _cgroup_gb() -> float:
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
    except Exception:
        return -1.0


def _report(tag: str, t0: float) -> None:
    print(f"[{tag}] {time.time() - t0:6.1f}s  rss={_peak_gb():5.1f}GB  cgroup={_cgroup_gb():5.1f}GB", flush=True)


# --------------------------------------------------------------------------
# phase 1: text encoder only
# --------------------------------------------------------------------------

def encode(args) -> None:
    import torch

    t0 = time.time()
    nodes = _boot()
    _report("boot", t0)

    clip = _call(
        nodes, "LTXAVTextEncoderLoader",
        text_encoder=args.text_encoder, ckpt_name=args.checkpoint, device="default",
    )[0]
    _report("encoder-loaded", t0)

    out = {
        "positive": _call(nodes, "CLIPTextEncode", clip=clip, text=args.positive)[0],
        "negative": _call(nodes, "CLIPTextEncode", clip=clip, text=args.negative)[0],
    }
    _report("encoded", t0)

    # Conditioning is a list of [tensor, dict]. Park it on the CPU in bf16: it
    # is read back while the 43 GB checkpoint is already resident, so every
    # gigabyte here comes straight off the headroom under the container cap.
    def pack(t):
        if not torch.is_tensor(t):
            return t
        return t.to("cpu", torch.bfloat16) if t.is_floating_point() else t.cpu()

    def to_cpu(cond):
        return [[pack(t), {k: pack(v) for k, v in d.items()}] for t, d in cond]

    torch.save({k: to_cpu(v) for k, v in out.items()}, args.conditioning)
    del clip, out
    gc.collect()
    _report("saved", t0)
    print(f"conditioning -> {args.conditioning} ({os.path.getsize(args.conditioning) / 2**20:.1f} MB)")


# --------------------------------------------------------------------------
# phase 2: diffusion checkpoint only
# --------------------------------------------------------------------------

def sample(args) -> None:
    """ComfyUI executes every graph under inference_mode, and the LTX VAE
    relies on it: decoding updates the sampler's output in place, which torch
    refuses across the inference-mode boundary."""
    import torch

    with torch.inference_mode():
        _sample(args)


def _sample(args) -> None:
    import numpy as np
    import torch
    from PIL import Image

    t0 = time.time()
    nodes = _boot()
    _report("boot", t0)

    model, _, vae = _call(nodes, "CheckpointLoaderSimple", ckpt_name=args.checkpoint)[:3]
    _report("checkpoint-loaded", t0)

    model = _call(nodes, "LoraLoaderModelOnly", model=model, lora_name=DISTILL_LORA,
                  strength_model=args.lora_strength)[0]
    audio_vae = _call(nodes, "LTXVAudioVAELoader", ckpt_name=args.checkpoint)[0]
    _report("lora+audio-vae", t0)

    cond = torch.load(args.conditioning, weights_only=False)
    positive, negative = _call(
        nodes, "LTXVConditioning",
        positive=cond["positive"], negative=cond["negative"], frame_rate=float(args.fps),
    )[:2]
    del cond
    gc.collect()

    plate = Image.open(args.image).convert("RGB")
    image = torch.from_numpy(np.array(plate).astype(np.float32) / 255.0)[None]
    image = _call(nodes, "LTXVPreprocess", image=image, img_compression=args.compression)[0]

    def av_pass(width, height, latent, sigmas, seed, pos, neg, audio_latent=None):
        """One concatenated audio-video denoise, returned separated again."""
        video = _call(nodes, "LTXVImgToVideoInplace", vae=vae, image=image, latent=latent,
                      strength=args.strength, bypass=False)[0]
        if audio_latent is None:
            audio_latent = _call(nodes, "LTXVEmptyLatentAudio", frames_number=args.frames,
                                 frame_rate=args.fps, batch_size=1, audio_vae=audio_vae)[0]
        av = _call(nodes, "LTXVConcatAVLatent", video_latent=video, audio_latent=audio_latent)[0]
        guider = _call(nodes, "CFGGuider", model=model, positive=pos, negative=neg, cfg=args.cfg)[0]
        out = _call(
            nodes, "SamplerCustomAdvanced",
            noise=_call(nodes, "RandomNoise", noise_seed=seed)[0],
            guider=guider,
            sampler=_call(nodes, "KSamplerSelect", sampler_name=args.sampler)[0],
            sigmas=_call(nodes, "ManualSigmas", sigmas=sigmas)[0],
            latent_image=av,
        )[0]
        return _call(nodes, "LTXVSeparateAVLatent", av_latent=out)[:2]

    base_latent = _call(nodes, "EmptyLTXVLatentVideo", width=args.width, height=args.height,
                        length=args.frames, batch_size=1)[0]
    video_latent, audio_latent = av_pass(
        args.width, args.height, base_latent, BASE_SIGMAS, args.seed, positive, negative
    )
    _report("base-pass", t0)

    if args.refine:
        upscaler = _call(nodes, "LatentUpscaleModelLoader", model_name=UPSCALER)[0]
        upsampled = _call(nodes, "LTXVLatentUpsampler", samples=video_latent,
                          upscale_model=upscaler, vae=vae)[0]
        # Guides from the base pass would be at the old resolution, so they are
        # cropped away before the refine pass reuses the conditioning.
        pos, neg, _ = _call(nodes, "LTXVCropGuides", positive=positive, negative=negative,
                            latent=video_latent)[:3]
        video_latent, audio_latent = av_pass(
            args.width * 2, args.height * 2, upsampled, REFINE_SIGMAS,
            args.seed + 1, pos, neg, audio_latent,
        )
        _report("refine-pass", t0)

    # The 22B transformer is still sitting in VRAM and the VAE needs room to
    # decode into. ComfyUI's own executor evicts models between nodes; calling
    # nodes directly skips that, so it has to be done by hand or the decode
    # dies with 2 GB free out of 48.
    import comfy.model_management as mm

    del model, positive, negative
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
    _report("models-evicted", t0)

    images = _call(nodes, "VAEDecodeTiled", samples=video_latent, vae=vae, tile_size=args.tile,
                   overlap=64, temporal_size=args.temporal_tile, temporal_overlap=4)[0]
    _report("video-decoded", t0)

    audio = _call(nodes, "LTXVAudioVAEDecode", samples=audio_latent, audio_vae=audio_vae)[0]
    _report("audio-decoded", t0)

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(images):
        Image.fromarray((frame.cpu().numpy() * 255).clip(0, 255).astype("uint8")).save(dest / f"{i:03d}.png")
    _write_wav(audio, dest / "audio.wav")

    meta = {
        "frames": len(images),
        "size": [images.shape[2], images.shape[1]],
        "fps": args.fps,
        "refined": bool(args.refine),
        "seed": args.seed,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": round(_peak_gb(), 1),
        "peak_cgroup_gb": round(_cgroup_gb(), 1),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def _write_wav(audio: dict, path: Path) -> None:
    """AUDIO is {waveform: [B, C, N] float, sample_rate: int}."""
    wav = audio["waveform"][0].cpu().float().clamp(-1, 1)
    pcm = (wav * 32767).to("cpu").short().numpy().T.tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(wav.shape[0])
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes(pcm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)

    e = sub.add_parser("encode", help="phase 1, text encoder only")
    e.add_argument("--text-encoder", default="gemma_3_12B_it.safetensors")
    e.add_argument("--checkpoint", default="ltx-2.3-22b-dev.safetensors")
    e.add_argument("--positive", required=True)
    e.add_argument("--negative", default="text, watermark, distorted product, morphing shape, "
                                         "jitter, cartoon, childish, ugly")
    e.add_argument("--conditioning", default="/workspace/dukaan/cond.pt")
    e.set_defaults(fn=encode)

    s = sub.add_parser("sample", help="phase 2, diffusion checkpoint only")
    s.add_argument("--checkpoint", default="ltx-2.3-22b-dev.safetensors")
    s.add_argument("--conditioning", default="/workspace/dukaan/cond.pt")
    s.add_argument("--image", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=640)
    s.add_argument("--frames", type=int, default=49, help="8n+1 for LTX temporal compression")
    s.add_argument("--fps", type=int, default=25)
    s.add_argument("--cfg", type=float, default=1.0)
    s.add_argument("--strength", type=float, default=0.7)
    s.add_argument("--compression", type=int, default=18)
    s.add_argument("--lora-strength", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--sampler", default="euler")
    s.add_argument("--tile", type=int, default=512, help="VAE decode tile size")
    s.add_argument("--temporal-tile", type=int, default=32, help="frames per VAE decode chunk")
    s.add_argument("--refine", action="store_true", help="second pass at 2x resolution")
    s.set_defaults(fn=sample)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

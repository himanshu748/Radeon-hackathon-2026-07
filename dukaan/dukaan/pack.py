"""One photo in, a shop's worth of creatives out.

The pipeline, end to end:

    photo -> cutout -> plate -> ONE GPU pass -> frames + audio
                                             |
                                             +-> pick a still per format
                                             |     -> compose type -> creatives
                                             +-> clip

The GPU runs once per pack, not once per format. That is not only a cost
decision: LTX-2.3 is the only generative model on the instance and it is a
video model, so a pass produces a moving scene rather than a picture. Stills
are lifted out of that scene, which is why three formats can look like three
different shots without three generations.

Type is composited afterwards and never generated. Models garble text, and a
seller's price and phone number are the two things on the creative that must be
exactly right.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .backend import Backend, Clip, _fit
from .config import Config
from .frames import fit_scene, pick_frames
from .layout import compose, contact_strip
from .spec import FORMATS, Brief, Format, Style


@dataclass
class PackResult:
    brief: Brief
    creatives: dict[str, Path] = field(default_factory=dict)
    frame_of: dict[str, int] = field(default_factory=dict)
    clip_frames: list[Path] = field(default_factory=list)
    clip: Path | None = None
    audio: Path | None = None
    backend: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{len(self.creatives)} creative(s) via {self.backend}"]
        for name, path in self.creatives.items():
            lines.append(f"  {name:7s} {path}  (frame {self.frame_of.get(name, '?')})")
        if self.clip:
            lines.append(f"  clip    {self.clip} ({len(self.clip_frames)} frames)")
        if self.audio:
            lines.append(f"  audio   {self.audio}")
        if self.meta.get("seconds"):
            lines.append(f"  gpu     {self.meta['seconds']}s on {self.backend}")
        return "\n".join(lines)


def make_plate(cut: Image.Image, style: Style, size: tuple[int, int]) -> Image.Image:
    """What the model is shown: the product, cleanly placed on the style wash.

    The plate is deliberately plain. Handing the model a finished-looking
    composition makes it redecorate; handing it the product on a flat ground
    leaves it the job of lighting a scene around something it must not change.
    """
    canvas = Image.new("RGB", size, style.backdrop)
    fitted = _fit(cut, size, band=0.0, fill=0.72)
    canvas.paste(fitted, (0, 0), fitted)
    return canvas


def build_pack(
    cfg: Config,
    backend: Backend,
    photo: Image.Image,
    brief: Brief,
    formats: tuple[Format, ...] = FORMATS,
    contact: str = "",
    write_clip: bool = True,
    on_step=None,
) -> PackResult:
    style = brief.styled()
    out_dir = cfg.out_dir / brief.product
    out_dir.mkdir(parents=True, exist_ok=True)
    result = PackResult(brief=brief, backend=backend.name)

    def step(msg: str) -> None:
        if on_step:
            on_step(msg)

    step("cutout")
    cut = backend.remove_background(photo)

    step(f"plate {cfg.width}x{cfg.height}")
    plate = make_plate(cut, style, cfg.size)
    plate.save(out_dir / f"{brief.style}_plate.png")

    step(f"render {cfg.frames} frames on {backend.name}")
    clip: Clip = backend.render(plate, style, cfg)
    result.meta = clip.meta

    step(f"pick {len(formats)} stills from {len(clip)} frames")
    picks = pick_frames(clip.frames, len(formats))

    for fmt, idx in zip(formats, picks):
        step(f"compose {fmt.name}")
        scene = fit_scene(clip.frames[idx], fmt.size)
        canvas = compose(scene, brief, fmt, style)
        canvas = contact_strip(canvas, contact, style)
        path = out_dir / f"{brief.style}_{fmt.name}.png"
        canvas.save(path)
        result.creatives[fmt.name] = path
        result.frame_of[fmt.name] = idx

    if write_clip:
        frame_dir = out_dir / f"{brief.style}_clip_frames"
        frame_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(clip.frames):
            p = frame_dir / f"{i:03d}.png"
            frame.save(p)
            result.clip_frames.append(p)
        result.clip = _write_gif(clip.frames, out_dir / f"{brief.style}_clip.gif", cfg.fps)
        if clip.audio:
            result.audio = out_dir / f"{brief.style}_clip.wav"
            result.audio.write_bytes(clip.audio)

    (out_dir / f"{brief.style}_manifest.json").write_text(
        json.dumps(
            {
                "product": brief.product,
                "headline": brief.headline,
                "subline": brief.subline,
                "style": brief.style,
                "backend": backend.name,
                "generation": {"width": cfg.width, "height": cfg.height,
                               "frames": cfg.frames, "fps": cfg.fps, "seed": cfg.seed,
                               "strength": cfg.strength, "refine": cfg.refine},
                "run": clip.meta,
                "creatives": {k: str(v) for k, v in result.creatives.items()},
                "still_from_frame": result.frame_of,
                "clip": str(result.clip) if result.clip else None,
                "audio": str(result.audio) if result.audio else None,
            },
            indent=2,
        )
    )
    return result


def _write_gif(frames: list[Image.Image], path: Path, fps: int) -> Path:
    """A GIF so the clip previews anywhere, including a GitHub README.

    mp4 is nicer and carries the audio, but needs ffmpeg on the box. The frames
    and the wav are written alongside, so a real encode is one command away.
    """
    if not frames:
        raise ValueError("no frames to write")
    head, *rest = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    head.save(path, save_all=True, append_images=rest,
              duration=max(int(1000 / max(fps, 1)), 20), loop=0, optimize=True)
    return path

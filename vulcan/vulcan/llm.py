"""Thin client for any OpenAI-compatible server (Ollama, vLLM, llama.cpp)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import Config


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    total_s: float


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=300,
        )

    def chat(self, messages: list[dict], stream: bool = False) -> ChatResult:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "stream": stream,
        }
        if self.cfg.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.cfg.enable_thinking}
        start = time.perf_counter()
        if not stream:
            r = self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            return ChatResult(
                text=data["choices"][0]["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                ttft_s=None,
                total_s=time.perf_counter() - start,
            )

        # Streaming: measure time-to-first-token for the benchmark harness.
        ttft = None
        chunks: list[str] = []
        n_tokens = 0
        with self._client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                import json as _json

                delta = _json.loads(line[6:])["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks.append(piece)
                    n_tokens += 1
        return ChatResult(
            text="".join(chunks),
            prompt_tokens=0,
            completion_tokens=n_tokens,
            ttft_s=ttft,
            total_s=time.perf_counter() - start,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._client.post("/embeddings", json={"model": self.cfg.embed_model, "input": texts})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

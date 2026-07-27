"""ReAct-style agent loop with JSON tool calls.

Uses a plain-text JSON protocol instead of native tool-calling so the same
loop works on every backend (Ollama, vLLM, llama.cpp) regardless of
tool-call support.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import tools
from .config import Config
from .llm import LLM
from .memory import Memory
from .rag import Index

SYSTEM = """You are Vulcan, a local developer-productivity agent. All inference runs on-device; never claim to access the internet.

You work in a loop. On each turn reply with EXACTLY ONE JSON object, nothing else:
  {"thought": "...", "tool": "<name>", "args": {...}}
or, when you have the answer:
  {"thought": "...", "final": "<answer for the user, cite files as path:line>"}

Strategy: start with search_code, it is almost always the fastest route. Answer as soon as you have enough evidence; do not keep exploring.

Tools:
- search_code {"query": str}            semantic search over the indexed codebase
- read_file   {"path": str, "start": int, "end": int}
- list_dir    {"path": str}
- grep        {"pattern": str, "glob": str}
- run_cmd     {"command": str}          allowlisted: tests, linters, git
- write_file  {"path": str, "content": str}
- remember    {"note": str}             persist a durable fact for future sessions

Prior session notes:
{memory}
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class Agent:
    def __init__(self, cfg: Config, root: Path, index_name: str = "default"):
        self.cfg = cfg
        self.root = root
        self.llm = LLM(cfg)
        self.index = Index(cfg, self.llm, index_name)
        self.memory = Memory(cfg, project=str(root))

    def _dispatch(self, tool: str, args: dict) -> str:
        if tool == "search_code":
            hits = self.index.search(args.get("query", ""))
            return "\n---\n".join(f"{h['path']}:{h['start']}-{h['end']} (score {h['score']:.2f})\n{h['text']}" for h in hits) or "no results"
        if tool == "read_file":
            return tools.read_file(self.root, args.get("path", ""), args.get("start", 1), args.get("end"))
        if tool == "list_dir":
            return tools.list_dir(self.root, args.get("path", "."))
        if tool == "grep":
            return tools.grep(self.root, args.get("pattern", ""), args.get("glob", "**/*"))
        if tool == "run_cmd":
            return tools.run_cmd(self.root, args.get("command", ""))
        if tool == "write_file":
            return tools.write_file(self.root, args.get("path", ""), args.get("content", ""))
        if tool == "remember":
            self.memory.remember(args.get("note", ""))
            return "noted"
        return f"ERROR: unknown tool {tool}"

    def run(self, task: str, on_step=None) -> str:
        notes = "\n".join(f"- {n}" for n in self.memory.recall()) or "(none)"
        messages = [
            {"role": "system", "content": SYSTEM.replace("{memory}", notes)},
            {"role": "user", "content": task},
        ]
        for _ in range(self.cfg.max_steps):
            reply = self.llm.chat(messages, stream=True).text
            match = JSON_RE.search(reply)
            if not match:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Reply with exactly one JSON object per the protocol."})
                continue
            try:
                step = json.loads(match.group())
            except json.JSONDecodeError as e:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Invalid JSON ({e}). Retry."})
                continue
            if on_step:
                on_step(step)
            if "final" in step:
                return step["final"]
            obs = self._dispatch(step.get("tool", ""), step.get("args", {}))
            messages.append({"role": "assistant", "content": match.group()})
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        return "Step limit reached without a final answer."

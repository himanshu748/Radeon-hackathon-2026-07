"""Agent tools. Each tool returns a string observation.

run_cmd is allowlisted: the agent can run tests and linters but cannot
execute arbitrary shell commands.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ALLOWED_CMDS = {"pytest", "python", "python3", "npm", "node", "cargo", "go", "ruff", "eslint", "tsc", "git"}
MAX_OBS = 6000


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OBS else s[:MAX_OBS] + f"\n...[truncated {len(s) - MAX_OBS} chars]"


def read_file(root: Path, path: str, start: int = 1, end: int | None = None) -> str:
    target = (root / path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return "ERROR: path escapes project root"
    if not target.is_file():
        return f"ERROR: no such file: {path}"
    lines = target.read_text(errors="ignore").splitlines()
    start = max(start, 1)
    end = end or min(len(lines), start + 200)
    body = "\n".join(f"{i}\t{l}" for i, l in enumerate(lines[start - 1 : end], start))
    return _clip(body)


def list_dir(root: Path, path: str = ".") -> str:
    target = (root / path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return "ERROR: path escapes project root"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return _clip("\n".join(entries))


def grep(root: Path, pattern: str, glob: str = "**/*") -> str:
    if glob in {"*", ".", ""}:
        glob = "**/*"
    out: list[str] = []
    for p in root.glob(glob):
        if p.is_file() and ".git" not in p.parts:
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if pattern in line:
                        out.append(f"{p.relative_to(root)}:{i}: {line.strip()}")
                        if len(out) >= 100:
                            return _clip("\n".join(out))
            except OSError:
                continue
    return _clip("\n".join(out) or "no matches")


def run_cmd(root: Path, command: str) -> str:
    argv = shlex.split(command)
    if not argv or Path(argv[0]).name not in ALLOWED_CMDS:
        return f"ERROR: command not in allowlist {sorted(ALLOWED_CMDS)}"
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 180s"
    return _clip(f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def write_file(root: Path, path: str, content: str) -> str:
    target = (root / path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return "ERROR: path escapes project root"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} chars to {path}"

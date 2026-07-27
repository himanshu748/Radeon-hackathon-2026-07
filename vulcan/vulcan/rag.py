"""Codebase RAG: chunk source files, embed via the GPU backend, search with cosine.

Storage is a single SQLite file with vectors as float32 blobs. No vector-DB
dependency, trivially portable between the dev laptop and the ROCm box.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .config import Config
from .llm import LLM

SOURCE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java", ".rb", ".c",
    ".cc", ".cpp", ".h", ".hpp", ".md", ".toml", ".yaml", ".yml", ".json",
    ".sh", ".sql", ".swift", ".kt",
}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".next", "target"}


def _venv_dirs(root: Path) -> set[Path]:
    """Virtualenvs named anything other than venv/.venv.

    Name matching alone silently indexes every package in site-packages,
    which turns `vulcan index .` into a multi-thousand-file embedding job.
    pyvenv.cfg is the definitive marker, so find those instead of guessing.
    """
    return {cfg.parent for cfg in root.rglob("pyvenv.cfg")}
CHUNK_LINES = 60
CHUNK_OVERLAP = 10
EMBED_BATCH = 32


class Index:
    def __init__(self, cfg: Config, llm: LLM, name: str = "default"):
        self.cfg = cfg
        self.llm = llm
        self.db = sqlite3.connect(cfg.data_dir / f"index-{name}.sqlite3")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            " id INTEGER PRIMARY KEY, path TEXT, start_line INTEGER,"
            " end_line INTEGER, text TEXT, vec BLOB)"
        )

    def build(self, root: Path) -> int:
        self.db.execute("DELETE FROM chunks")
        batch_texts: list[str] = []
        batch_meta: list[tuple[str, int, int]] = []
        count = 0
        skip_roots = _venv_dirs(root)
        for path in sorted(root.rglob("*")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if any(p in skip_roots for p in path.parents):
                continue
            if not path.is_file() or path.suffix not in SOURCE_EXTS:
                continue
            try:
                lines = path.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            step = CHUNK_LINES - CHUNK_OVERLAP
            for start in range(0, max(len(lines), 1), step):
                seg = lines[start : start + CHUNK_LINES]
                if not any(s.strip() for s in seg):
                    continue
                rel = str(path.relative_to(root))
                batch_texts.append(f"{rel}\n" + "\n".join(seg))
                batch_meta.append((rel, start + 1, start + len(seg)))
                if len(batch_texts) >= EMBED_BATCH:
                    count += self._flush(batch_texts, batch_meta)
                    batch_texts, batch_meta = [], []
        if batch_texts:
            count += self._flush(batch_texts, batch_meta)
        self.db.commit()
        return count

    def _flush(self, texts: list[str], meta: list[tuple[str, int, int]]) -> int:
        vecs = self.llm.embed(texts)
        for (rel, s, e), text, vec in zip(meta, texts, vecs):
            arr = np.asarray(vec, dtype=np.float32)
            arr /= np.linalg.norm(arr) or 1.0
            self.db.execute(
                "INSERT INTO chunks (path, start_line, end_line, text, vec) VALUES (?,?,?,?,?)",
                (rel, s, e, text, arr.tobytes()),
            )
        return len(texts)

    def search(self, query: str, k: int = 6) -> list[dict]:
        rows = self.db.execute("SELECT path, start_line, end_line, text, vec FROM chunks").fetchall()
        if not rows:
            return []
        q = np.asarray(self.llm.embed([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        mat = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        scores = mat @ q
        top = np.argsort(-scores)[:k]
        return [
            {"path": rows[i][0], "start": rows[i][1], "end": rows[i][2], "text": rows[i][3], "score": float(scores[i])}
            for i in top
        ]

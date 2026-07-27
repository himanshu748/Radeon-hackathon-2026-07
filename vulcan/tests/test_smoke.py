import json
from pathlib import Path

from vulcan import bench, rag, tools
from vulcan.config import Config
from vulcan.llm import ChatResult


def test_read_file_blocks_escape(tmp_path: Path):
    assert "escapes" in tools.read_file(tmp_path, "../secrets.txt")


def test_run_cmd_allowlist(tmp_path: Path):
    assert "not in allowlist" in tools.run_cmd(tmp_path, "rm -rf /")


def test_write_and_read_roundtrip(tmp_path: Path):
    tools.write_file(tmp_path, "a/b.txt", "hello")
    assert "hello" in tools.read_file(tmp_path, "a/b.txt")


def test_grep(tmp_path: Path):
    (tmp_path / "x.py").write_text("needle = 1\n")
    assert "x.py:1" in tools.grep(tmp_path, "needle")


class _StubLLM:
    """Returns a fixed-size streamed response without touching a backend."""

    def __init__(self, cfg):
        self.cfg = cfg

    def chat(self, messages, stream=False):
        return ChatResult(text="x", prompt_tokens=0, completion_tokens=10, ttft_s=0.1, total_s=1.1)


def test_run_concurrency_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "LLM", _StubLLM)
    out = bench.run_concurrency(Config(), "t", levels=(1, 2), out_dir=tmp_path)
    assert set(out["runs"]) == {"concurrency_1", "concurrency_2"}
    two = out["runs"]["concurrency_2"]
    assert two["requests"] == 2
    # 2 requests x 10 chunks, so aggregate must exceed per-request
    assert two["aggregate_chunks_per_s"] > two["per_request_chunks_per_s"]
    assert json.loads((tmp_path / "t.json").read_text())["axis"] == "concurrency"


def test_run_prefill_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "LLM", _StubLLM)
    out = bench.run_prefill(Config(), "p", word_counts=(8, 16), out_dir=tmp_path)
    assert set(out["runs"]) == {"prefill_8w", "prefill_16w"}
    assert out["runs"]["prefill_8w"]["input_words"] == 8
    assert json.loads((tmp_path / "p.json").read_text())["axis"] == "prefill"


def test_venv_dirs_detects_oddly_named_env(tmp_path: Path):
    odd = tmp_path / ".myenv"
    (odd / "lib").mkdir(parents=True)
    (odd / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (tmp_path / "src").mkdir()
    assert rag._venv_dirs(tmp_path) == {odd}


def test_bench_compare(tmp_path: Path):
    base = {"label": "laptop", "runs": {
        "short": {"ttft_s_median": 1.0, "chunks_per_s_median": 20.0},
        "medium": {"ttft_s_median": 1.5, "chunks_per_s_median": 18.0},
        "long": {"ttft_s_median": 2.0, "chunks_per_s_median": 15.0}}}
    cand = {"label": "radeon", "runs": {
        "short": {"ttft_s_median": 0.25, "chunks_per_s_median": 60.0},
        "medium": {"ttft_s_median": 0.4, "chunks_per_s_median": 54.0}}}
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(json.dumps(base))
    c.write_text(json.dumps(cand))
    out = bench.compare(b, c)
    assert "3.00x" in out
    assert out.count("|") > 10
    assert "n/a" in out

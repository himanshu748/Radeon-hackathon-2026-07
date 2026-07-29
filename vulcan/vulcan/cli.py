"""CLI: vulcan index / chat / ask / bench."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from . import bench as bench_mod
from .agent import Agent
from .config import load
from .llm import LLM
from .rag import Index

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()



def _masked(url: str) -> str:
    """Show enough of the endpoint to identify it, not enough to reuse it.

    The instance id is the whole path to someone else's GPU budget. It ends up
    in screenshots, screen recordings and CI logs, so the banner prints the host
    and elides the rest.
    """
    head, _, tail = url.rstrip("/").rpartition("/spaces/")
    if not head:
        return url
    return f"{head}/spaces/<instance>/" + tail.split("/", 1)[-1]


@app.command()
def index(path: Path = typer.Argument(..., exists=True), name: str = "default") -> None:
    """Index a codebase for semantic search."""
    cfg = load()
    idx = Index(cfg, LLM(cfg), name)
    with console.status(f"Indexing {path} ..."):
        n = idx.build(path.resolve())
    console.print(f"[green]Indexed {n} chunks from {path}[/green]")


@app.command()
def ask(question: str, root: Path = Path("."), name: str = "default") -> None:
    """One-shot agent run against an indexed codebase."""
    cfg = load()
    agent = Agent(cfg, root.resolve(), name)

    def show(step: dict) -> None:
        if "tool" in step:
            console.print(f"[dim]→ {step['tool']} {step.get('args', {})}[/dim]")

    answer = agent.run(question, on_step=show)
    console.print(Panel(answer, title="Vulcan", border_style="cyan"))


@app.command()
def chat(root: Path = Path("."), name: str = "default") -> None:
    """Interactive multi-turn session."""
    cfg = load()
    agent = Agent(cfg, root.resolve(), name)
    console.print(f"[cyan]Vulcan[/cyan] on {_masked(cfg.base_url)} · model {cfg.model} · ctrl-d to exit")
    while True:
        try:
            question = console.input("[bold]you>[/bold] ")
        except EOFError:
            break
        if not question.strip():
            continue
        answer = agent.run(question, on_step=lambda s: console.print(f"[dim]→ {s.get('tool', 'final')}[/dim]"))
        console.print(Panel(answer, border_style="cyan"))


@app.command()
def bench(label: str = "local", repeats: int = 3) -> None:
    """Measure TTFT and generation speed on the current backend."""
    cfg = load()
    console.print(f"Benchmarking {cfg.model} at {_masked(cfg.base_url)} ...")
    results = bench_mod.run(cfg, label, repeats)
    for name, run in results["runs"].items():
        console.print(f"  {name:7s} ttft={run['ttft_s_median']}s  speed={run['chunks_per_s_median']} chunks/s")
    console.print(f"[green]Saved bench-results/{label}.json[/green]")


@app.command("bench-concurrency")
def bench_concurrency(label: str = "concurrency", levels: str = "1,2,4,8") -> None:
    """Aggregate throughput as concurrent requests scale."""
    cfg = load()
    parsed = tuple(int(x) for x in levels.split(",") if x.strip())
    results = bench_mod.run_concurrency(cfg, label, parsed)
    for name, run in results["runs"].items():
        console.print(
            f"  {name:15s} agg={run['aggregate_chunks_per_s']} chunks/s  "
            f"per-req={run['per_request_chunks_per_s']}  ttft={run['ttft_s_median']}s"
        )
    console.print(f"[green]Saved bench-results/{label}.json[/green]")


@app.command("bench-prefill")
def bench_prefill(label: str = "prefill", words: str = "128,512,2048,8192") -> None:
    """Time-to-first-token against input length."""
    cfg = load()
    parsed = tuple(int(x) for x in words.split(",") if x.strip())
    results = bench_mod.run_prefill(cfg, label, parsed)
    for name, run in results["runs"].items():
        console.print(f"  {name:15s} ttft={run['ttft_s_median']}s")
    console.print(f"[green]Saved bench-results/{label}.json[/green]")


@app.command("bench-compare")
def bench_compare(baseline: Path, candidate: Path) -> None:
    """Markdown comparison table between two `vulcan bench` result files."""
    console.print(bench_mod.compare(baseline, candidate))


if __name__ == "__main__":
    app()

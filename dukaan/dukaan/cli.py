"""CLI: dukaan pack / styles / doctor."""
from __future__ import annotations

from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from .backend import RadeonBackend, make_backend
from .config import load
from .instance import InstanceError
from .pack import build_pack
from .spec import FORMATS, FORMATS_BY_NAME, STYLES, Brief

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _masked(url: str) -> str:
    """Show enough of the instance URL to identify it, not enough to use it.

    The tunnel in front of the instance takes no credentials, so the URL is the
    credential: anyone who reads one off a screenshot or a CI log can spend the
    GPU budget behind it. The host is useful to see, the instance id is not.
    """
    head, _, tail = url.rstrip("/").rpartition("/")
    return f"{head}/{tail[:6]}..." if len(tail) > 6 else url


@app.command()
def styles() -> None:
    """Looks a seller can pick without writing a prompt."""
    table = Table(title="Styles")
    table.add_column("name")
    table.add_column("what it looks like")
    for s in STYLES:
        table.add_row(s.name, s.prompt.split(",")[0])
    console.print(table)

    fmt_table = Table(title="Formats")
    for col in ("name", "size", "used for"):
        fmt_table.add_column(col)
    for f in FORMATS:
        fmt_table.add_row(f.name, f"{f.width}x{f.height}", f.note)
    console.print(fmt_table)


@app.command()
def doctor() -> None:
    """Say plainly which backend a run would use, before it costs anything."""
    cfg = load()
    if cfg.offline:
        console.print("[yellow]No DUKAAN_INSTANCE set.[/yellow] Runs use the mock CPU backend.")
        console.print("Point it at the Radeon instance for real generation:")
        console.print("  [dim]export DUKAAN_INSTANCE=https://<host>/instances/<instance-id>[/dim]")
        raise typer.Exit(0)

    console.print(f"instance: {_masked(cfg.instance)}")
    backend = RadeonBackend(cfg)
    try:
        # rocm-smi cannot reach libdrm inside this container and torch returns
        # an empty device name, so the architecture, the PCI model id and the
        # compute-unit count stand in. They are what the driver will actually
        # tell you, and they identify the part without guessing at its
        # marketing name.
        report = backend.inst.run(
            "import subprocess, os\n"
            "def sh(c):\n"
            "    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()\n"
            "print(sh(\"/opt/venv/bin/python3 -c \\\"import torch;p=torch.cuda.get_device_properties(0);\"\n"
            "         \"print('arch:', p.gcnArchName);print('vram_gb:', round(p.total_memory/2**30,1));\"\n"
            "         \"print('compute_units:', p.multi_processor_count);\"\n"
            "         \"print('torch:', torch.__version__)\\\"\"))\n"
            "print('card_model:', sh(\"rocm-smi --showproductname 2>/dev/null | \"\n"
            "      \"grep -i 'card model' | head -1 | awk '{print $NF}'\") or 'unknown')\n"
            "print('container_ram_cap_gb:', round(int(open('/sys/fs/cgroup/memory.max').read())/2**30, 1))\n"
            "print('runner:', 'yes' if os.path.exists('/workspace/%s/radeon_ltx.py') else 'MISSING')\n"
            % cfg.remote_dir,
            timeout=180,
        )
    except InstanceError as exc:
        console.print(f"[red]unreachable:[/red] {exc}")
        raise typer.Exit(1)

    for line in report.strip().splitlines():
        console.print(f"  {line}")
    if "MISSING" in report:
        console.print("[yellow]runner not deployed.[/yellow] Run [bold]dukaan deploy[/bold] first.")
        raise typer.Exit(1)
    console.print("[green]ready[/green]")


@app.command()
def deploy() -> None:
    """Copy the GPU runner onto the instance's persistent volume."""
    cfg = load()
    if cfg.offline:
        raise typer.BadParameter("set DUKAAN_INSTANCE first")
    runner = Path(__file__).resolve().parent.parent / "scripts" / "radeon_ltx.py"
    if not runner.exists():
        raise typer.BadParameter(f"runner not found at {runner}")
    RadeonBackend(cfg).deploy(runner)
    console.print(f"[green]deployed[/green] {runner.name} -> /workspace/{cfg.remote_dir}/")


@app.command()
def pack(
    photo: Path = typer.Argument(..., exists=True, help="One product photo"),
    headline: str = typer.Option(..., "--headline", "-h", help="The line that sells it"),
    subline: str = typer.Option("", "--subline", "-s", help="Price, offer, or detail"),
    style: str = typer.Option("studio", "--style", help="See `dukaan styles`"),
    product: str = typer.Option("", "--product", help="Folder name for the output"),
    contact: str = typer.Option("", "--contact", help="Phone or handle, rendered exactly"),
    formats: str = typer.Option("", "--formats", help="Comma list, default all"),
    no_clip: bool = typer.Option(False, "--no-clip", help="Skip writing the clip and its frames"),
) -> None:
    """Build a campaign pack from one photo."""
    cfg = load()
    chosen = FORMATS
    if formats:
        names = [n.strip() for n in formats.split(",") if n.strip()]
        unknown = [n for n in names if n not in FORMATS_BY_NAME]
        if unknown:
            raise typer.BadParameter(f"unknown format(s): {', '.join(unknown)}")
        chosen = tuple(FORMATS_BY_NAME[n] for n in names)

    brief = Brief(
        product=product or photo.stem,
        headline=headline,
        subline=subline,
        style=style,
    )
    try:
        brief.styled()
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    backend = make_backend(cfg)
    if cfg.offline:
        console.print("[yellow]mock backend[/yellow] (no DUKAAN_INSTANCE), output is a layout preview")

    image = Image.open(photo)
    with console.status("building pack ..."):
        result = build_pack(
            cfg,
            backend,
            image,
            brief,
            formats=chosen,
            contact=contact,
            write_clip=not no_clip,
            on_step=lambda m: console.print(f"[dim]  {m}[/dim]"),
        )
    console.print(result.summary())


if __name__ == "__main__":
    app()

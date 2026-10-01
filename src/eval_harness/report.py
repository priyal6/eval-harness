"""Terminal tables and CSV output."""

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from .runner import Result
from .store import RESULTS_DIR

console = Console()


def short(model: str) -> str:
    return model.split("/")[-1].removesuffix(":free")


def show_grid(results: list[Result], tests: list[dict], models: list[str]) -> None:
    table = Table(title="Per test")
    table.add_column("Test")
    table.add_column("Expected")
    for m in models:
        table.add_column(short(m), overflow="fold")

    cells = {(r.test_id, r.model): r for r in results}
    for test in tests:
        row = [test["id"], str(test.get("expected", "(rubric)"))]
        for m in models:
            r = cells[(test["id"], m)]
            if r.error.startswith("skipped"):
                row.append("[dim]skipped[/dim]")
            elif r.error:
                row.append("[yellow]error[/yellow]")
            else:
                mark = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
                row.append(f"{mark} {r.answer[:20]}")
        table.add_row(*row)
    console.print(table)


def show_summary(results: list[Result], models: list[str]) -> None:
    """Accuracy only counts tests the model actually answered."""
    table = Table(title="Scoreboard")
    table.add_column("Model", no_wrap=True)
    for col in ("Correct", "Accuracy", "Avg time (s)", "Avg output tokens", "Errors", "Skipped"):
        table.add_column(col, justify="right")

    rows = []
    for model in models:
        rs = [r for r in results if r.model == model]
        answered = [r for r in rs if not r.error]
        skipped = sum(r.error.startswith("skipped") for r in rs)
        errors = len(rs) - len(answered) - skipped
        passed = sum(r.passed for r in answered)
        acc = passed / len(answered) if answered else 0
        times = [r.latency_s for r in answered if r.latency_s is not None]
        toks = [r.completion_tokens for r in answered if r.completion_tokens is not None]
        rows.append((acc, passed, model, len(answered), times, toks, errors, skipped))

    for acc, passed, model, n, times, toks, errors, skipped in sorted(rows, reverse=True):
        table.add_row(
            short(model), f"{passed}/{n}", f"{acc:.0%}" if n else "-",
            f"{mean(times):.1f}" if times else "-",
            f"{mean(toks):.0f}" if toks else "-",
            str(errors or ""), str(skipped or ""),
        )
    console.print(table)


def write_csv(results: list[Result]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    return path
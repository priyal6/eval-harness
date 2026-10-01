"""Command line: list-free, ping, run."""

import argparse

from dotenv import load_dotenv
from rich.table import Table

from .client import ask, list_free_models, make_client
from .config import load_config, load_tests
from .judge import make_judge_client
from .report import console, show_grid, show_summary, write_csv
from .runner import plan, run
from .store import Cache, Usage


def cmd_list_free(args) -> None:
    try:
        models = list_free_models()
    except Exception as e:
        raise SystemExit(f"Couldn't reach OpenRouter: {e}")
    table = Table(title=f"{len(models)} free text models right now")
    table.add_column("Model ID (paste into config.yaml)")
    table.add_column("Name")
    table.add_column("Context", justify="right")
    for m in models:
        table.add_row(m["id"], m["name"], f"{m['context']:,}" if m["context"] else "-")
    console.print(table)
    console.print("[dim]'stealth/' models are anonymous and usually log prompts.[/dim]")


def cmd_ping(args) -> None:
    cfg = load_config(args.config)
    models = args.models or cfg.models
    usage = Usage()
    if usage.remaining(cfg.settings.daily_request_limit) < len(models):
        raise SystemExit("Not enough requests left today to ping every model.")
    client = make_client(cfg.settings)
    msg = [{"role": "user", "content": "Reply with just the word OK."}]
    for model in models:
        usage.add()
        try:
            reply = ask(client, model, msg, cfg.settings, retries=0)
            console.print(f"[green]✓[/green] {model}  ({reply.latency_s}s): {reply.text.strip()[:40]!r}")
        except Exception as e:
            console.print(f"[red]✗[/red] {model}: {type(e).__name__}: {str(e)[:150]}")
    console.print(f"[dim]Requests used today: {usage.count}/{cfg.settings.daily_request_limit}[/dim]")


def cmd_run(args) -> None:
    cfg = load_config(args.config)
    if args.models:
        cfg.models = args.models
    tests = load_tests(args.tests)
    cache, usage = Cache(), Usage()
    limit = cfg.settings.daily_request_limit

    jobs = plan(cfg, tests, cache)
    new = sum(1 for *_, cached in jobs if not cached)
    left = usage.remaining(limit)
    console.print(f"{len(tests)} tests × {len(cfg.models)} models = {len(jobs)} answers "
                  f"([green]{len(jobs) - new} cached[/green], [bold]{new} new requests[/bold]).")
    console.print(f"OpenRouter requests left today: {left}/{limit}")
    console.print(f"Judge: {cfg.judge_model} grades all {len(jobs)} answers (OpenAI, a few cents at most)")

    if args.dry_run:
        return
    if new > left:
        console.print(f"[yellow]Only {left} requests left, so {new - left} answers will be "
                      "skipped. Remove some tests/models, or try again tomorrow.[/yellow]")
    if not args.yes and input("Go ahead? [y/N] ").strip().lower() != "y":
        return

    judge_client = make_judge_client()
    client = make_client(cfg.settings) if new else None
    done = [0]

    def progress(r):
        done[0] += 1
        mark = "✓" if r.passed else ("!" if r.error else "✗")
        note = f"  [yellow]{r.error}[/yellow]" if r.error else f"  [dim]{r.judge_reason}[/dim]"
        console.print(f"[{done[0]}/{len(jobs)}] {mark} {r.test_id} · {r.model}{note}")

    results = run(cfg, tests, client, judge_client, cache, usage, on_result=progress)
    console.print()
    show_grid(results, tests, cfg.models)
    show_summary(results, cfg.models)
    path = write_csv(results)
    console.print(f"Full answers and judge reasons saved to [bold]{path}[/bold]. "
                  f"OpenRouter requests used today: {usage.count}/{limit}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="eval-harness",
                                     description="Compare OpenRouter models on your own tests.")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-free", help="show the free models available right now")

    p = sub.add_parser("ping", help="check each model answers (1 request per model)")
    p.add_argument("--models", nargs="+", help="override the models in config.yaml")

    r = sub.add_parser("run", help="run the tests")
    r.add_argument("--tests", default="tests.yaml")
    r.add_argument("--models", nargs="+", help="override the models in config.yaml")
    r.add_argument("--dry-run", action="store_true", help="show the request count and stop")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")

    args = parser.parse_args()
    {"list-free": cmd_list_free, "ping": cmd_ping, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
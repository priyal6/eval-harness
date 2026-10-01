"""Load config.yaml and tests.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Settings:
    temperature: float = 0
    max_tokens: int = 4000
    reasoning_effort: str | None = "low"
    timeout_seconds: float = 180
    seconds_between_requests: float = 3.2
    daily_request_limit: int = 50


@dataclass
class Config:
    models: list[str]
    judge_model: str
    settings: Settings
    system_prompt: str


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not raw.get("models"):
        raise SystemExit(f"No models listed in {path}.")
    return Config(
        models=list(raw["models"]),
        judge_model=raw.get("judge_model", "gpt-6-luna"),
        settings=Settings(**(raw.get("settings") or {})),
        system_prompt=(raw.get("system_prompt") or "").strip(),
    )


def load_tests(path: str | Path) -> list[dict]:
    tests = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    for t in tests:
        if "id" not in t or "prompt" not in t:
            raise SystemExit(f"Every test needs an 'id' and a 'prompt': {t}")
        if "expected" not in t and "rubric" not in t:
            raise SystemExit(f"Test '{t['id']}' needs an 'expected' answer or a 'rubric'.")
    return tests
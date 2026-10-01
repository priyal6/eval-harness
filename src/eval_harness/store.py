"""Response cache and daily request counter, both stored in results/."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results")


def cache_key(model: str, system_prompt: str, prompt: str, settings) -> str:
    payload = json.dumps([model, system_prompt, prompt, settings.temperature,
                          settings.max_tokens, settings.reasoning_effort])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Cache:
    """Saved model answers, so re-runs don't spend requests."""

    def __init__(self, path: Path = RESULTS_DIR / "cache.jsonl"):
        self.path = path
        self.items: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.items[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, record: dict) -> None:
        record = {"key": key, **record}
        self.items[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Usage:
    """Counts requests sent today (UTC, when OpenRouter's daily limit resets)."""

    def __init__(self, path: Path = RESULTS_DIR / "usage.json"):
        self.path = path
        self.today = datetime.now(timezone.utc).date().isoformat()
        data = json.loads(path.read_text()) if path.exists() else {}
        self.count = data.get("requests", 0) if data.get("date") == self.today else 0

    def remaining(self, limit: int) -> int:
        return max(0, limit - self.count)

    def add(self, n: int = 1) -> None:
        self.count += n
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"date": self.today, "requests": self.count}))
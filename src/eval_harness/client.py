import os 
import time
from dataclasses import dataclass

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .config import Settings
BASE_URL = "https://openrouter.ai/api/v1"

class DailyLimitReached(Exception):
    """OpenRouter says the free-model quota for today is used up."""

@dataclass
class Reply:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None

def make_client(settings: Settings) -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: 
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and paste your key in."
        )
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=settings.timeout_seconds,
        max_retries=0,
        default_headers={"X-Title": "eval-harness"},
    )

def ask(client: OpenAI, model: str, messages: list[dict], settings: Settings,
        retries: int = 2) -> Reply:
    """Send one chat request, retrying short-term failures."""
    extra = {}
    if settings.reasoning_effort:
        extra["reasoning"] = {"effort": settings.reasoning_effort}

    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                extra_body=extra or None,
            )
        except RateLimitError as e:
            if "per-day" in str(e).lower() or "per day" in str(e).lower():
                raise DailyLimitReached(str(e)) from e
            if attempt == retries:
                raise
            time.sleep(15 * (attempt + 1))  # per-minute limit: back off and retry
            continue
        except (APITimeoutError, APIConnectionError):
            if attempt == retries:
                raise
            time.sleep(5)
            continue
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries:  # provider hiccup
                time.sleep(5)
                continue
            raise

        latency = time.perf_counter() - start
        if not resp.choices:  # OpenRouter sometimes returns an error in a 200 body
            raise RuntimeError(f"No answer returned: {getattr(resp, 'error', resp)}")
        choice = resp.choices[0]
        usage = resp.usage
        return Reply(
            text=choice.message.content or "",
            latency_s=round(latency, 2),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )
    raise RuntimeError("unreachable")

def list_free_models() -> list[dict]:
    """Current free text models from OpenRouter's public catalogue (no key needed)."""
    data = httpx.get(f"{BASE_URL}/models", timeout=30).raise_for_status().json()["data"]
    free = []
    for m in data:
        pricing = m.get("pricing") or {}
        is_free = m["id"].endswith(":free") or (
            pricing.get("prompt") == "0" and pricing.get("completion") == "0"
        )
        outputs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
        if is_free and "text" in outputs:
            free.append({
                "id": m["id"],
                "name": m.get("name", ""),
                "context": m.get("context_length"),
            })
    return sorted(free, key=lambda m: m["id"])

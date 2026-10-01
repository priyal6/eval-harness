"""Runs every test against every model, then asks the judge to grade each answer."""

import time
from dataclasses import dataclass

from .client import DailyLimitReached, ask
from .config import Config
from .judge import final_answer, judge
from .store import Cache, Usage, cache_key

MAX_FAILURES_PER_MODEL = 2   # stop wasting requests on a model that isn't answering


@dataclass
class Result:
    model: str
    test_id: str
    passed: bool = False
    answer: str = ""
    judge_reason: str = ""
    latency_s: float | None = None
    completion_tokens: int | None = None
    cached: bool = False
    error: str = ""
    response: str = ""


def plan(cfg: Config, tests: list[dict], cache: Cache):
    """Every (test, model) pair, and whether its answer is already cached."""
    jobs = []
    for test in tests:              # test-major: if quota runs out, models stay comparable
        for model in cfg.models:
            key = cache_key(model, cfg.system_prompt, test["prompt"], cfg.settings)
            jobs.append((test, model, key, cache.get(key)))
    return jobs


def get_answer(cfg, test, model, client):
    messages = [{"role": "user", "content": test["prompt"]}]
    if cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    return ask(client, model, messages, cfg.settings)


def run(cfg: Config, tests: list[dict], client, judge_client, cache: Cache,
        usage: Usage, on_result=lambda r: None) -> list[Result]:
    results, failures, dead = [], {}, set()
    out_of_quota, first_live = False, True

    for test, model, key, cached in plan(cfg, tests, cache):
        r = Result(model=model, test_id=test["id"])

        # 1. Get the model's answer (from the cache, or by asking it)
        if cached:
            r.response, r.cached = cached["text"], True
            r.latency_s, r.completion_tokens = cached.get("latency_s"), cached.get("completion_tokens")
        elif model in dead:
            r.error = "skipped: model not answering"
        elif out_of_quota or usage.remaining(cfg.settings.daily_request_limit) == 0:
            out_of_quota, r.error = True, "skipped: daily limit reached"
        else:
            if not first_live:
                time.sleep(cfg.settings.seconds_between_requests)
            first_live = False
            try:
                usage.add()
                reply = get_answer(cfg, test, model, client)
                failures[model] = 0
                r.response, r.latency_s, r.completion_tokens = (
                    reply.text, reply.latency_s, reply.completion_tokens)
                cache.put(key, {"model": model, "test_id": test["id"], "text": reply.text,
                                "latency_s": reply.latency_s,
                                "completion_tokens": reply.completion_tokens})
            except DailyLimitReached:
                out_of_quota, r.error = True, "skipped: daily limit reached"
            except Exception as e:  # one bad model shouldn't stop the run
                failures[model] = failures.get(model, 0) + 1
                if failures[model] >= MAX_FAILURES_PER_MODEL:
                    dead.add(model)
                r.error = f"{type(e).__name__}: {str(e)[:200]}"

        # 2. Ask the judge whether it's correct
        if not r.error:
            r.answer = final_answer(r.response)
            try:
                r.passed, r.judge_reason = judge(judge_client, cfg.judge_model, test, r.response)
            except Exception as e:
                r.error = f"judge failed: {type(e).__name__}: {str(e)[:150]}"

        results.append(r)
        on_result(r)
    return results
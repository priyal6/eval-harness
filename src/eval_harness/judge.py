"""GPT judge: asks OpenAI whether a model's answer is correct."""

import os
import re

from openai import OpenAI

JUDGE_PROMPT = """You are grading an AI model's answer.

Question:
{question}

{reference}

Model's answer:
{answer}

Judge only the final answer, not the working or style. Equivalent forms are
fine (e.g. "5 cents" = "$0.05"). If it gives several different answers or no
clear answer, it is incorrect.

Reply in exactly this format:
VERDICT: CORRECT or INCORRECT
REASON: <one short sentence>"""


def make_judge_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=key)


def final_answer(text: str) -> str:
    """The model's 'Answer: ...' line, for showing in the results table."""
    matches = re.findall(r"answer\s*:\s*(.+)", text or "", re.IGNORECASE)
    return matches[-1].strip(" *.") if matches else (text or "").strip()[-40:]


def judge(client: OpenAI, model: str, test: dict, answer: str) -> tuple[bool, str]:
    """Returns (is_correct, reason)."""
    if "expected" in test:
        reference = f"Correct answer:\n{test['expected']}"
    else:
        reference = f"What counts as correct:\n{test['rubric']}"

    prompt = JUDGE_PROMPT.format(question=test["prompt"], reference=reference,
                                 answer=answer[-6000:] or "(empty)")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    correct = "VERDICT: CORRECT" in text.upper()
    reason = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    return correct, reason.group(1).strip() if reason else text.strip()
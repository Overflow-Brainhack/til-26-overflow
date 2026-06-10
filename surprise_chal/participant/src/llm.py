"""Bare-minimal LLM helper for the LLM agent template.

One function — `call_llm(system, user)` — POSTs an OpenAI-compatible chat
request to OpenRouter and returns the assistant's text (or None on any error).
It reads your key from the OPENROUTER_API_KEY environment variable.

This is intentionally tiny and dependency-light so you can see exactly how the
interface works. Add streaming, retries, caching, multi-call planning, etc. — whatever you want. The only hard rule is
the turn deadline (default 10s): if you don't answer in time, your turn is a
no-op, so keep total work inside `timeout`.
"""

from __future__ import annotations

import json
import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Pick any model your key can reach. Cheap+fast is usually the right trade for a
# 10s-per-turn game. Override without editing code via the OPENROUTER_MODEL env var.
MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")


async def call_llm(
    system: str,
    user: str,
    *,
    model: str = MODEL,
    max_tokens: int = 600,
    timeout: float = 9.0,
) -> str | None:
    """Return the model's text reply, or None on missing key / error / timeout.

    `timeout` is total wall-clock for the call — keep it under the turn deadline.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        # Ask for raw JSON back so it's easy to parse into actions.
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def parse_json(text: str | None) -> dict:
    """Best-effort: pull a JSON object out of the model's reply. {} on failure.

    Tolerates ```json fences and leading/trailing prose so one stray token never
    costs you the whole turn.
    """
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        # fall back to the first {...} block
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i : j + 1])
            except Exception:
                return {}
        return {}

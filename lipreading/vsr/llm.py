"""LLM correction of a noisy lip-reading transcript (any OpenAI-compatible API).

The visual model confuses sounds that look the same on the lips. The LLM gets
its reading (and, when available, what the other person had just said) and
proposes the most likely sentences. In the app, the user taps the right one.

Defaults to Groq; set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY to use another provider.
"""
import json
import os
import time
import urllib.request
from typing import Callable, List, Optional

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

# Pilot 01 ran with the Spanish-language version of this prompt.
SYSTEM = """You are the decoder of a Spanish lip-reading system for people who have lost their voice.
You receive the NOISY transcript produced by a visual model. That model confuses sounds that look the same on the lips:
p/b/m, f/v, t/d/n/l, k/g/j, s/z/c, ch/y/ll, and sometimes drops or invents short words.
Your task: propose the {n} sentences the person MOST LIKELY said, from most to least likely, in Spanish.
Rules:
- Keep the meaning and shape of the reading; fix what is visually confusable.
- If there is context (what they were just told), use it: the sentence should be a natural reply.
- Short, natural sentences, the way a person would say them at home.
- Answer ONLY with JSON: {{"options": ["sentence 1", "sentence 2", ...]}}"""

Ask = Callable[[List[dict]], str]


def _ask(messages: List[dict], attempts: int = 4) -> str:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set LLM_API_KEY (or GROQ_API_KEY)")
    body = json.dumps({"model": MODEL, "temperature": 0.2, "response_format": {"type": "json_object"},
                       "messages": messages}).encode()
    for i in range(attempts):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     # Groq's CDN rejects (403) the default "Python-urllib" user agent
                     "User-Agent": "lip-reading-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:  # 429 / network: retry with backoff
            if i == attempts - 1:
                raise
            time.sleep(3 * (i + 1))
    raise AssertionError("unreachable")


def options(reading: str, context: Optional[str] = None, n: int = 3, ask: Optional[Ask] = None) -> List[str]:
    """Up to n candidate sentences; falls back to the raw reading if the LLM fails."""
    ask = ask or _ask
    user = f'Visual model reading: "{reading}"'
    if context:
        user = f'They were just told: "{context}"\n' + user
    try:
        data = json.loads(ask([{"role": "system", "content": SYSTEM.format(n=n)}, {"role": "user", "content": user}]))
        opts = [o.strip() for o in data.get("options", []) if isinstance(o, str) and o.strip()]
    except Exception as e:
        print(f"  [LLM] failed: {e}")
        opts = []
    return (opts or [reading])[:n]

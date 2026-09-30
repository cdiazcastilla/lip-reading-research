"""Evaluation metrics for lip reading.

WER (word error rate) = word-level edit distance / number of reference words.
0 % is perfect. Lip reading cannot see accents or punctuation, so sentences are
compared after normalisation: lowercase, no accents, no punctuation.
"""
import re
import unicodedata
from typing import Iterable, Sequence

# A sentence with WER <= 25 % is counted as "understandable": a listener can
# usually recover the meaning with one or two wrong words.
UNDERSTANDABLE_WER = 0.25


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # strip accents (ñ -> n too)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance between two token sequences (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, ignoring spaces."""
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    return edit_distance(ref, hyp) / max(1, len(ref))


def summarize(values: Iterable[float]) -> dict:
    """Mean WER plus the share of exact and understandable sentences."""
    values = list(values)
    if not values:
        return {"n": 0, "wer": None, "exact": None, "understandable": None}
    n = len(values)
    return {
        "n": n,
        "wer": sum(values) / n,
        "exact": sum(v == 0 for v in values) / n,
        "understandable": sum(v <= UNDERSTANDABLE_WER for v in values) / n,
    }

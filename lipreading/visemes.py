"""Spanish viseme mapping: compare sentences by what the lips can actually show.

Many sounds look identical on the lips (p/b/m, f/v, t/d/n/l/s...). Mapping text
to viseme classes lets us ask a fairer question than WER: did the model see
the right mouth shapes, even if it picked the wrong words?

This is a coarse, rule-based mapping from Spanish orthography. It is meant for
analysis, not as a phonetic transcription.
"""
from typing import List

from .metrics import edit_distance, normalize

# Classes: P bilabial, F labiodental, T alveolar/dental, K velar, Y palatal,
# vowels A (open), E (spread: e, i), O (rounded), U (rounded, closed).
_DIGRAPHS = [("ch", "Y"), ("ll", "Y"), ("rr", "T"), ("qu", "K"), ("gu", "K")]
_LETTER = {
    **{c: "P" for c in "pbm"},
    **{c: "F" for c in "fv"},
    **{c: "T" for c in "tdnlrsczxy"},
    **{c: "K" for c in "kgjq"},
    "h": "",  # silent in Spanish
    "w": "U",
    "a": "A", "e": "E", "i": "E", "o": "O", "u": "U",
}


def to_visemes(text: str) -> List[str]:
    """Text -> list of viseme classes. Repeated classes collapse into one,
    because consecutive identical mouth shapes are not visually separable."""
    s = normalize(text).replace(" ", "")
    out: List[str] = []
    i = 0
    while i < len(s):
        pair = s[i:i + 2]
        digraph = next((v for d, v in _DIGRAPHS if pair == d), None)
        if digraph:
            v, i = digraph, i + 2
        else:
            v, i = _LETTER.get(s[i], s[i].upper()), i + 1
        if v and not (out and out[-1] == v):
            out.append(v)
    return out


def viseme_similarity(a: str, b: str) -> float:
    """1 - normalised edit distance between viseme sequences (1.0 = same mouth shapes)."""
    va, vb = to_visemes(a), to_visemes(b)
    return 1 - edit_distance(va, vb) / max(len(va), len(vb), 1)


def rerank_by_visemes(reading: str, candidates: List[str]) -> List[str]:
    """Order candidate sentences by how well they match the mouth shapes of the
    visual model's reading. Stable for ties, so the LLM's own order breaks them."""
    return sorted(candidates, key=lambda c: -viseme_similarity(reading, c))

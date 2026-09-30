"""Pilot 01 analysis: word, character and viseme-level agreement.

    python experiments/pilot-01/analyze.py

Prints the summary table and writes viseme_vs_wer.png next to this file.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lipreading.metrics import cer, summarize, wer  # noqa: E402
from lipreading.visemes import viseme_similarity  # noqa: E402


def pct(x):
    return f"{x * 100:.0f} %"


def main():
    clips = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))["clips"]
    n = len(clips)
    stages = {"Visual model alone": "reading", "+ LLM with context (1st option)": "llm_ctx_top1"}

    print(f"{n} sentences, silent speech, one speaker\n")
    print(f"{'Stage':34} {'WER':>6} {'CER':>6} {'Understandable':>15} {'Viseme match':>13}")
    for name, key in stages.items():
        s = summarize(wer(c["sentence"], c[key]) for c in clips)
        c_rate = sum(cer(c["sentence"], c[key]) for c in clips) / n
        vis = sum(viseme_similarity(c["sentence"], c[key]) for c in clips) / n
        print(f"{name:34} {pct(s['wer']):>6} {pct(c_rate):>6} {pct(s['understandable']):>15} {pct(vis):>13}")

    closer = sum(viseme_similarity(c["sentence"], c["reading"]) > viseme_similarity(c["sentence"], c["llm_ctx_top1"]) for c in clips)
    better = sum(wer(c["sentence"], c["llm_ctx_top1"]) < wer(c["sentence"], c["reading"]) for c in clips)
    worse = sum(wer(c["sentence"], c["llm_ctx_top1"]) > wer(c["sentence"], c["reading"]) for c in clips)
    print(f"\nThe LLM improved {better}, worsened {worse}, left {n - better - worse} unchanged (by WER).")
    print(f"In {closer}/{n} sentences the raw reading matches the true mouth shapes better than the LLM's answer.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping the figure)")
        return
    x = [viseme_similarity(c["sentence"], c["reading"]) for c in clips]
    y = [1 - min(1.0, wer(c["sentence"], c["reading"])) for c in clips]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.scatter(x, y, s=28, color="#2a6fdb", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.plot([0, 1], [0, 1], color="#9aa3b0", lw=1, ls="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Viseme similarity to the true sentence")
    ax.set_ylabel("Word accuracy (1 − WER, floored at 0)")
    ax.set_title("The model sees the mouth shapes, but loses the words", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(HERE, "viseme_vs_wer.png")
    fig.savefig(out)
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()

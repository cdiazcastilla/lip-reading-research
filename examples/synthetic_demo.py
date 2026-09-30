"""Train the personal lip reader on synthetic clips and test it on new ones.

    python examples/synthetic_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lipreading.personal import PersonalLipReader, describe_clip  # noqa: E402
from lipreading.personal.synthetic import PHRASES, UNKNOWN, make_clip  # noqa: E402


def main(seed: int = 7, per_phrase: int = 5, tests_per_phrase: int = 10):
    rng = np.random.default_rng(seed)
    train = [(p, describe_clip(make_clip(p, rng))) for _ in range(per_phrase) for p in PHRASES]
    reader = PersonalLipReader()
    stats = reader.train(train)

    print(f"Trained on {len(train)} clips, {len(PHRASES)} phrases")
    print(f"Leave-one-out accuracy: {stats['loo_accuracy']:.0%}")
    op = stats["operating_point"]
    print(f"Calibrated threshold {op['threshold']:.3f}, margin {op['margin']:.2f}")

    decisions = {"match": 0, "ambiguous": 0, "reject": 0}
    correct = 0
    for p in PHRASES:
        for _ in range(tests_per_phrase):
            r = reader.classify(describe_clip(make_clip(p, rng)))
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
            correct += r.decision == "match" and r.phrase == p
    n = len(PHRASES) * tests_per_phrase
    print(f"New clips: {n} | matched {decisions['match']} | ambiguous {decisions['ambiguous']} | "
          f"rejected {decisions['reject']} | correct matches {correct}/{decisions['match'] or 1}")

    # A phrase the user never recorded: speaking a wrong phrase is worse than silence.
    unknown = list(UNKNOWN)[0]
    results = [reader.classify(describe_clip(make_clip(unknown, rng))) for _ in range(tests_per_phrase)]
    spoken = sum(r.decision == "match" for r in results)
    print(f"Unrecorded phrase \"{unknown}\": wrongly spoken {spoken}/{len(results)} times")


if __name__ == "__main__":
    main()

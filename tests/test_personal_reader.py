import numpy as np
import pytest

from lipreading.personal import PersonalLipReader, describe_clip, describe_frame, dtw
from lipreading.personal.synthetic import PHRASES, UNKNOWN, make_clip


def test_descriptor_is_pose_invariant():
    rng = np.random.default_rng(0)
    frame = make_clip("Thank you", rng, noise=0)[20]
    base = describe_frame(frame)
    # rotate, scale and move the whole face
    a, s, t = 0.3, 1.7, np.array([0.2, -0.1])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = {k: np.hstack([(v[:, :2] @ rot.T) * s + t, v[:, 2:] * s]) for k, v in frame.items()}
    other = describe_frame(moved)
    assert np.allclose(base.v, other.v, atol=1e-9)
    assert base.ap == pytest.approx(other.ap)


def test_incomplete_frame_is_rejected():
    assert describe_frame({"upper_out": np.zeros((11, 3))}) is None


def test_dtw_is_zero_on_identical_and_tolerates_speed_changes():
    t = np.linspace(0, 1, 40)[:, None]
    x = np.hstack([np.sin(6 * t), np.cos(6 * t)])
    slow = np.hstack([np.sin(6 * t ** 1.3), np.cos(6 * t ** 1.3)])
    other = np.hstack([np.cos(9 * t), np.sin(2 * t)])
    assert dtw(x, x, 0.25) == 0
    assert dtw(x, slow, 0.25) < dtw(x, other, 0.25)


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(1)
    reader = PersonalLipReader()
    stats = reader.train([(p, describe_clip(make_clip(p, rng))) for _ in range(5) for p in PHRASES])
    return reader, stats, rng


def test_training_calibrates_thresholds(trained):
    reader, stats, _ = trained
    assert reader.ready
    assert stats["loo_accuracy"] >= 0.9
    assert set(stats["phrases"]) == set(PHRASES)
    assert reader.threshold > 0 and 0.6 <= reader.margin <= 0.85


def test_recognises_new_recordings(trained):
    reader, _, rng = trained
    for p in PHRASES:
        r = reader.classify(describe_clip(make_clip(p, rng)))
        assert r.phrase == p


def test_stays_silent_on_an_unrecorded_phrase(trained):
    reader, _, rng = trained
    unknown = list(UNKNOWN)[0]
    matches = [reader.classify(describe_clip(make_clip(unknown, rng))).decision == "match" for _ in range(10)]
    assert sum(matches) <= 1


def test_needs_two_phrases():
    rng = np.random.default_rng(2)
    reader = PersonalLipReader()
    stats = reader.train([("Thank you", describe_clip(make_clip("Thank you", rng))) for _ in range(4)])
    assert not reader.ready and stats["reason"] == "need_at_least_two_phrases"
    assert reader.classify([]).decision == "not_ready"

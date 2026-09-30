"""Personal lip reader: recognises a user's own phrases from face landmarks.

Each user records a few clips of every phrase they want to say. The reader is
trained only on those clips, so it learns that person's face and articulation.

Pipeline
    frame (raw MediaPipe landmarks, grouped into zones)
      -> face-aligned descriptor (lips + chin + depth), invariant to
         translation, in-plane rotation and distance to the camera
      -> PCA fitted on THIS user's frames (their personal shape space)
      -> silence trimming + smoothing + velocities (deltas)
      -> resampling to a fixed length
      -> k-nearest-neighbour DTW against the user's own templates

Acceptance thresholds are not hard-coded: they are calibrated with
leave-one-out over the user's dataset (each clip is classified against the
rest). The same pass gives an estimated accuracy per phrase, which tells the
user which phrase needs more recordings. Design rule: a wrong phrase spoken
aloud is worse than silence, so errors are penalised more than misses.

This is a NumPy port of the reader that runs in the browser in DeeprVoice.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULTS = dict(
    pca_dims=10,             # components of the personal shape space
    resample_len=32,         # fixed sequence length for DTW
    band=0.25,               # Sakoe-Chiba band, as a fraction of the length
    delta_ratio=0.5,         # weight of velocity vs position
    z_weight=0.5,            # MediaPipe depth is noisy: weigh it less
    k_nearest=3,             # templates averaged per phrase
    max_per_phrase=12,       # most recent templates used per phrase
    min_per_phrase=3,        # recordings needed for a phrase to count
    loo_queries_per_phrase=6,
    wrong_penalty=3.0,       # in calibration, one error costs three hits
    ambiguous_factor=1.5,    # up to threshold * 1.5 the result is "ambiguous"
)

ZONES = ("upper_out", "upper_in", "lower_out", "lower_in", "chin", "cheek_l", "cheek_r")


@dataclass
class FrameDescriptor:
    v: np.ndarray    # shape vector in the face frame
    ap: float        # inner-lip opening / face width (used to detect speech activity)
    w: float         # mouth-corner width / face width


def describe_frame(zones: Dict[str, np.ndarray], z_weight: float = DEFAULTS["z_weight"]) -> Optional[FrameDescriptor]:
    """Landmark zones of one frame -> descriptor in the face's own coordinate frame.

    `zones` maps each name in ZONES to an (n, 2) or (n, 3) array of normalised
    MediaPipe coordinates; the four lip contours have 11 points each, ordered
    from the left mouth corner to the right one (see landmarks.py).
    """
    try:
        uo, ui, lo, li, chin = (np.asarray(zones[k], float) for k in ZONES[:5])
        cl, cr = np.asarray(zones["cheek_l"], float), np.asarray(zones["cheek_r"], float)
    except KeyError:
        return None
    if min(len(uo), len(ui), len(lo), len(li)) < 11:
        return None

    # Face frame: origin between the cheeks, x axis from left to right cheek,
    # unit = face width. Removes translation, in-plane rotation and scale.
    c_l, c_r = cl[:, :2].mean(0), cr[:, :2].mean(0)
    origin, axis = (c_l + c_r) / 2, c_r - c_l
    s = float(np.hypot(*axis))
    if not s > 1e-6:
        return None
    cos, sin = axis / s
    rot = np.array([[cos, sin], [-sin, cos]])

    # Mouth corners repeat in the upper and lower contours: skip them in the lower ones.
    pts = np.vstack([uo[:11], lo[1:10], ui[:11], li[1:10], chin])
    xy = ((pts[:, :2] - origin) @ rot.T) / s

    # Protrusion: depth of the outer lips relative to the chin, in face widths.
    outer = np.vstack([uo[:11], lo[1:10]])
    if outer.shape[1] > 2:
        z = z_weight * (outer[:, 2] - chin[:, 2].mean()) / s
    else:
        z = np.zeros(len(outer))

    v = np.concatenate([xy.ravel(), z])
    if not np.all(np.isfinite(v)):
        return None
    ap = float(np.hypot(*(ui[5, :2] - li[5, :2])) / s)
    w = float(np.hypot(*(uo[0, :2] - uo[10, :2])) / s)
    return FrameDescriptor(v, ap, w)


def describe_clip(frames: Sequence[Dict[str, np.ndarray]], z_weight: float = DEFAULTS["z_weight"]) -> List[FrameDescriptor]:
    return [d for d in (describe_frame(f, z_weight) for f in frames) if d is not None]


# ── Signal processing ────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean, components [k, D], eigenvalues [k]) of the frame covariance."""
    mu = X.mean(0)
    cov = np.cov(X - mu, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:k]
    keep = vals[order] > 1e-14
    return mu, vecs[:, order[keep]].T, vals[order[keep]]


def moving_average(x: np.ndarray, half: int = 2) -> np.ndarray:
    """Centred moving average over time (axis 0), edges replicated."""
    pad = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    if x.ndim == 1:
        return np.convolve(pad, kernel, mode="valid")
    return np.stack([np.convolve(pad[:, d], kernel, mode="valid") for d in range(x.shape[1])], 1)


def _shrinking_average(a: np.ndarray, half: int = 2) -> np.ndarray:
    """Moving average whose window shrinks at the edges (no padding)."""
    return np.array([a[max(0, i - half): i + half + 1].mean() for i in range(len(a))])


def trim_range(ap: np.ndarray, w: np.ndarray, k: float = 0.15, pad: int = 3) -> Tuple[int, int]:
    """First and last frame of speech, from mouth opening, width and their motion.

    Training clips (push-to-talk) and live utterances have different edges;
    trimming both the same way is what makes them comparable.
    """
    T = len(ap)
    if T < 8:
        return 0, T - 1
    motion = np.concatenate([[0], np.abs(np.diff(ap)) + 0.7 * np.abs(np.diff(w))])
    offset = (ap - ap.min()) + 0.5 * np.abs(w - np.sort(w)[T // 2])
    act = _shrinking_average(offset) + 3 * _shrinking_average(motion)
    lo, hi = act.min(), act.max()
    if not hi - lo > 1e-9:
        return 0, T - 1
    idx = np.flatnonzero(act > lo + k * (hi - lo))
    if len(idx) == 0 or idx[-1] - idx[0] < 6:
        return 0, T - 1
    return max(0, idx[0] - pad), min(T - 1, idx[-1] + pad)


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation of a [T, D] sequence to n frames."""
    t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, d]) for d in range(x.shape[1])], 1)


def deltas(x: np.ndarray) -> np.ndarray:
    """Central-difference velocity, edges replicated."""
    p = np.pad(x, [(1, 1), (0, 0)], mode="edge")
    return (p[2:] - p[:-2]) / 2


def dtw(a: np.ndarray, b: np.ndarray, band: float) -> float:
    """Dynamic Time Warping with a Sakoe-Chiba band and Euclidean frame cost,
    normalised by the total length so short and long clips are comparable."""
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m) + 1, 2)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


# ── Reader ───────────────────────────────────────────────────────────────────

@dataclass
class Result:
    decision: str                      # "match" | "ambiguous" | "reject" | "too_short" | "not_ready"
    phrase: Optional[str] = None
    score: Optional[float] = None
    ratio: Optional[float] = None
    top3: List[Tuple[str, float]] = field(default_factory=list)


class PersonalLipReader:
    def __init__(self, **options):
        self.o = {**DEFAULTS, **options}
        self.ready = False
        self.templates: List[Tuple[str, np.ndarray]] = []
        self.stats: dict = {}

    def train(self, clips: Sequence[Tuple[str, List[FrameDescriptor]]]) -> dict:
        """clips: (phrase, frame descriptors), most recent first."""
        o = self.o
        by_phrase: Dict[str, List[List[FrameDescriptor]]] = defaultdict(list)
        for phrase, descs in clips:
            if len(descs) >= 8 and len(by_phrase[phrase]) < o["max_per_phrase"]:
                by_phrase[phrase].append(descs)
        used = {p: l for p, l in by_phrase.items() if len(l) >= o["min_per_phrase"]}
        self.ready, self.templates = False, []
        self.stats = {"counts": {p: len(l) for p, l in by_phrase.items()}, "phrases": {}}
        if len(used) < 2:
            self.stats["reason"] = "need_at_least_two_phrases"
            return self.stats

        # 1. Personal shape space
        frames = np.array([d.v for l in used.values() for descs in l for d in descs])
        self.mu, self.components, vals = fit_pca(frames, o["pca_dims"])
        self.scale = float(np.sqrt(vals[0])) if len(vals) else 1.0

        # 2. Position sequences, and how much velocity should weigh
        raw = [(p, x) for p, l in used.items() for descs in l if (x := self._positions(descs)) is not None]
        var_pos = sum(float((x ** 2).sum()) for _, x in raw)
        var_del = sum(float((deltas(x) ** 2).sum()) for _, x in raw)
        self.delta_w = np.sqrt(o["delta_ratio"] * var_pos / var_del) if var_del > 0 else 1.0
        self.templates = [(p, self._combine(x)) for p, x in raw]

        # 3. Leave-one-out calibration
        self._calibrate()
        self.ready = True
        return self.stats

    def _positions(self, descs: List[FrameDescriptor]) -> Optional[np.ndarray]:
        if len(descs) < 8:
            return None
        s, e = trim_range(np.array([d.ap for d in descs]), np.array([d.w for d in descs]))
        cut = descs[s:e + 1]
        if len(cut) < 5:
            return None
        proj = (np.array([d.v for d in cut]) - self.mu) @ self.components.T / self.scale
        return moving_average(proj)

    def _combine(self, x: np.ndarray) -> np.ndarray:
        return resample(np.hstack([x, deltas(x) * self.delta_w]), self.o["resample_len"])

    def sequence(self, descs: List[FrameDescriptor]) -> Optional[np.ndarray]:
        x = self._positions(descs)
        return None if x is None else self._combine(x)

    def rank(self, seq: np.ndarray, exclude: int = -1) -> List[Tuple[str, float]]:
        """Phrases ordered by the mean of their k smallest DTW distances."""
        dists: Dict[str, List[float]] = defaultdict(list)
        for i, (phrase, tpl) in enumerate(self.templates):
            if i != exclude:
                dists[phrase].append(dtw(seq, tpl, self.o["band"]))
        scores = [(p, float(np.mean(sorted(d)[: self.o["k_nearest"]]))) for p, d in dists.items()]
        return sorted(scores, key=lambda t: t[1])

    def _calibrate(self):
        o = self.o
        queries: Dict[str, List[int]] = defaultdict(list)
        for i, (p, _) in enumerate(self.templates):
            if len(queries[p]) < o["loo_queries_per_phrase"]:
                queries[p].append(i)
        res = []
        for idx in (i for l in queries.values() for i in l):
            r = self.rank(self.templates[idx][1], exclude=idx)
            if len(r) >= 2:
                truth = self.templates[idx][0]
                res.append((truth, r[0][0], r[0][1], r[0][1] / (r[1][1] + 1e-12)))

        # Top-1 accuracy per phrase: what the app shows next to each phrase
        per = defaultdict(lambda: [0, 0])
        for truth, pred, _, _ in res:
            per[truth][0] += 1
            per[truth][1] += truth == pred
        self.stats["phrases"] = {p: {"accuracy": ok / n, "tests": n} for p, (n, ok) in per.items()}
        correct = [r for r in res if r[0] == r[1]]
        self.stats["loo_accuracy"] = len(correct) / len(res) if res else None

        # Operating point: maximise accepted hits - penalty * accepted errors
        # over a grid of (distance threshold, best/second-best ratio).
        base = np.sort([r[2] for r in (correct or res)])
        grid = []
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            for mult in (1.0, 1.1, 1.25):
                thr = float(np.quantile(base, q)) * mult
                for margin in np.arange(0.6, 1.0001, 0.05):
                    obj = sum((1 if t == p else -o["wrong_penalty"]) for t, p, d1, ratio in res
                              if d1 <= thr and ratio <= margin)
                    grid.append((obj, thr, float(margin)))
        best = max(g[0] for g in grid)
        # Well-separated data ties many points; take the middle of the optimal
        # region, not its most permissive edge: the real world is noisier.
        optimal = [g for g in grid if g[0] >= best - 1e-9]
        self.threshold = float(sorted(g[1] for g in optimal)[(len(optimal) - 1) // 2])
        self.margin = min(0.85, float(sorted(g[2] for g in optimal)[(len(optimal) - 1) // 2]))
        accepted = [r for r in res if r[2] <= self.threshold and r[3] <= self.margin]
        self.stats["operating_point"] = {
            "threshold": self.threshold,
            "margin": self.margin,
            "accepted": len(accepted) / len(res) if res else 0.0,
            "precision_when_accepted": (sum(r[0] == r[1] for r in accepted) / len(accepted)) if accepted else None,
        }

    def classify(self, descs: List[FrameDescriptor]) -> Result:
        if not self.ready:
            return Result("not_ready")
        seq = self.sequence(descs)
        if seq is None:
            return Result("too_short")
        r = self.rank(seq)
        d1 = r[0][1]
        ratio = d1 / ((r[1][1] if len(r) > 1 else np.inf) + 1e-12)
        if d1 <= self.threshold and ratio <= self.margin:
            decision = "match"
        elif d1 <= self.threshold * self.o["ambiguous_factor"]:
            decision = "ambiguous"
        else:
            decision = "reject"
        return Result(decision, r[0][0], d1, ratio, r[:3])

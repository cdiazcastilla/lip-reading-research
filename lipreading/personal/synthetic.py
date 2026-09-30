"""Synthetic mouth-landmark clips, for tests and demos without real faces.

Real recordings are biometric data and stay private. This generator draws a
parametric mouth (opening, width and protrusion over time) and places it on a
face with random head position, in-plane rotation, distance to the camera,
speaking rate and landmark noise, so the reader's invariances can be tested.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Each phrase = keyframes of (opening, width, protrusion), in face widths.
# Rough analogues of vowel/consonant mouth shapes.
PHRASES: Dict[str, Sequence[Tuple[float, float, float]]] = {
    "I need water": [(0.02, 0.40, 0.00), (0.10, 0.36, 0.02), (0.03, 0.44, 0.00), (0.12, 0.30, 0.05), (0.02, 0.40, 0.00)],
    "Thank you": [(0.02, 0.40, 0.00), (0.06, 0.46, 0.00), (0.09, 0.32, 0.06), (0.02, 0.40, 0.00)],
    "Call my daughter": [(0.02, 0.40, 0.00), (0.14, 0.38, 0.01), (0.01, 0.42, 0.00), (0.08, 0.30, 0.05), (0.11, 0.44, 0.00), (0.02, 0.40, 0.00)],
    "I am in pain": [(0.02, 0.40, 0.00), (0.15, 0.40, 0.00), (0.00, 0.40, 0.00), (0.05, 0.47, 0.00), (0.02, 0.40, 0.00)],
}

# Never used for training: the reader should stay silent on it.
UNKNOWN: Dict[str, Sequence[Tuple[float, float, float]]] = {
    "Open the window": [(0.02, 0.40, 0.00), (0.09, 0.28, 0.07), (0.00, 0.38, 0.00), (0.13, 0.46, 0.00), (0.06, 0.26, 0.06), (0.02, 0.40, 0.00)],
}


def _contour(width: float, lift: float, n: int = 11) -> np.ndarray:
    """Arc from the left corner to the right one; lift = height at the centre."""
    x = np.linspace(-width / 2, width / 2, n)
    y = lift * (1 - (2 * x / width) ** 2)
    return np.stack([x, y], 1)


def _face(opening: float, width: float, protrusion: float) -> Dict[str, np.ndarray]:
    """Mouth + chin + cheeks in a canonical face frame (face width = 1, y down)."""
    mouth_y, thick = 0.25, 0.05
    upper_in = _contour(width * 0.9, -opening / 2) + [0, mouth_y]
    lower_in = _contour(width * 0.9, opening / 2) + [0, mouth_y]
    upper_out = _contour(width, -opening / 2 - thick) + [0, mouth_y]
    lower_out = _contour(width, opening / 2 + thick) + [0, mouth_y]
    chin = _contour(0.5, 0.10, 10) + [0, 0.45 + opening * 0.6]
    cheek = np.stack([np.cos(np.linspace(0, 2 * np.pi, 22)) * 0.05, np.sin(np.linspace(0, 2 * np.pi, 22)) * 0.05], 1)
    zones = {
        "upper_out": upper_out, "upper_in": upper_in, "lower_out": lower_out, "lower_in": lower_in,
        "chin": chin, "cheek_l": cheek + [-0.5, 0.05], "cheek_r": cheek + [0.5, 0.05],
    }
    depth = {"upper_out": -protrusion, "lower_out": -protrusion, "upper_in": -protrusion / 2,
             "lower_in": -protrusion / 2, "chin": 0.0, "cheek_l": 0.02, "cheek_r": 0.02}
    return {k: np.hstack([v, np.full((len(v), 1), depth[k])]) for k, v in zones.items()}


def make_clip(phrase: str, rng: np.random.Generator, noise: float = 0.004, pad_frames: int = 10) -> List[Dict[str, np.ndarray]]:
    """One recording of a phrase: rest -> keyframes -> rest, with random pose and speed."""
    keys = np.array({**PHRASES, **UNKNOWN}[phrase])
    frames_per_key = rng.integers(6, 11)                      # speaking rate
    t = np.linspace(0, len(keys) - 1, frames_per_key * (len(keys) - 1))
    track = np.stack([np.interp(t, np.arange(len(keys)), keys[:, d]) for d in range(3)], 1)
    rest = np.repeat(keys[:1], pad_frames, 0)
    track = np.vstack([rest, track, rest])

    angle = rng.normal(0, 0.12)                               # head roll, radians
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(0.18, 0.35)                           # distance to the camera
    shift = rng.uniform(0.3, 0.7, 2)                          # position in the image
    clip = []
    for opening, width, protrusion in track:
        frame = {}
        for k, pts in _face(opening, width, protrusion).items():
            xy = (pts[:, :2] @ rot.T) * scale + shift + rng.normal(0, noise * scale, (len(pts), 2))
            frame[k] = np.hstack([xy, pts[:, 2:] * scale])
        clip.append(frame)
    return clip

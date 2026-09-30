"""MediaPipe FaceLandmarker (478-point mesh) indices used by the personal reader."""
from typing import Dict

import numpy as np

# Lip contours, each ordered from the left mouth corner (61/78) to the right one (291/308).
UPPER_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LOWER_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
CHIN = [17, 314, 405, 321, 375, 152, 148, 176, 149, 150]
CHEEK_L = [116, 123, 147, 213, 192, 214, 210, 211, 32, 36, 50, 187, 205, 206, 207, 216, 212, 202, 204, 194, 142, 129]
CHEEK_R = [345, 352, 376, 433, 411, 425, 426, 427, 436, 432, 422, 424, 418, 262, 266, 280, 411, 434, 430, 431, 371, 358]


def zones_from_mesh(mesh: np.ndarray) -> Dict[str, np.ndarray]:
    """(478, 3) array of normalised MediaPipe landmarks -> zones for describe_frame()."""
    mesh = np.asarray(mesh, float)
    return {
        "upper_out": mesh[UPPER_OUT],
        "upper_in": mesh[UPPER_IN],
        "lower_out": mesh[LOWER_OUT],
        "lower_in": mesh[LOWER_IN],
        "chin": mesh[CHIN],
        "cheek_l": mesh[CHEEK_L],
        "cheek_r": mesh[CHEEK_R],
    }

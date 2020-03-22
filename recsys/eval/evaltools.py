from typing import Sequence

import numpy as np


def rmse(x: Sequence[float], y: Sequence[float]) -> float:
    sum_squares = np.sum(np.power(np.asarray(x) - np.asarray(y), 2))
    return np.sqrt(float(sum_squares) / len(x))

from typing import Sequence

import numpy as np


class CombinedModel(object):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray, y: Sequence[float], p: int, q: int):
        pass

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass
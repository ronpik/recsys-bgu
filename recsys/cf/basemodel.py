from typing import Sequence, NamedTuple

import numpy as np
from scipy.sparse import spmatrix


class BaseModel(object):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    def __init__(self):
        self.n_users = None
        self.n_items = None

    def fit(self, data: np.ndarray, y: Sequence[float], p: int, q: int):
        pass

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass


class SVDModelParams(NamedTuple):
    users_bias: np.ndarray
    items_bias: np.ndarray
    users_latent_features: np.ndarray
    items_latent_features: np.ndarray


def initialize_parameters(users_items_matrix: spmatrix) -> SVDModelParams:
    num_users =





from typing import Sequence, NamedTuple

import numpy as np
from scipy.sparse import spmatrix


class BaseModel(object):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    def __init__(self):
        self.n_users: int = None
        self.n_items: int = None

        self.model_parameters_: SVDModelParams = None

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
    num_users, num_items = users_items_matrix.shape
    users_mean_rating = np.mean(users_items_matrix, axis=1)
    items_mean_rating = np.mean(users_items_matrix, axis=0)








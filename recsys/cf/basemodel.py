from typing import Sequence, NamedTuple

import numpy as np
from scipy.sparse import spmatrix, find

INITIALIZE_LATENT_FEATURES_SCALE = 0.005


class BaseModel(object):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    def __init__(self):
        self.n_users: int = None
        self.n_items: int = None
        self.user_latent_dim: int = None
        self.item_latent_dim: int = None

        self.model_parameters_: SVDModelParams = None

    def fit(self, data: spmatrix, p: int, q: int):
        self.n_users, self.n_items = data.shape
        self.user_latent_dim = p
        self.item_latent_dim = q
        self.model_parameters_ = initialize_parameters(data, p, q)

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass


class SVDModelParams(NamedTuple):
    mean_rating: float
    users_bias: np.ndarray
    items_bias: np.ndarray
    users_latent_features: np.ndarray
    items_latent_features: np.ndarray


def initialize_parameters(users_items_matrix: spmatrix, user_latent_dim: int, item_latent_dim: int) -> SVDModelParams:
    user, item, rating = find(users_items_matrix)

    user_non_zero_count = np.bincount(user)
    user_non_zero_sum = np.bincount(user, weights=rating)
    users_mean_rating = user_non_zero_sum / user_non_zero_count

    item_non_zero_count = np.bincount(item)
    item_non_zero_sum = np.bincount(item, weights=rating)
    items_mean_rating = item_non_zero_sum / item_non_zero_count

    mean_rating = users_items_matrix.sum() / len(rating)

    scale = INITIALIZE_LATENT_FEATURES_SCALE
    latent_user_features = np.random.normal(0, scale, user_latent_dim * item_latent_dim)\
        .reshape(item_latent_dim, user_latent_dim)
    latent_item_features = np.random.normal(0, scale, user_latent_dim * item_latent_dim)\
        .reshape(user_latent_dim, item_latent_dim)

    return SVDModelParams(mean_rating,
                          users_mean_rating,
                          items_mean_rating,
                          latent_user_features,
                          latent_item_features
                          )







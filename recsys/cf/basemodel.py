import os
import json
import time
from math import sqrt
from random import shuffle, Random, sample
from typing import Sequence, NamedTuple, List, Tuple
from itertools import islice

import pandas as pd
import numpy as np
import tqdm as tqdm

from recsys.eval.evaltools import rmse
from recsys.utils.data.yelp_dataset import split_dataset
from recsys.cf import SVDModelEngine, AbstractSVDModelParams

INITIALIZE_LATENT_FEATURES_SCALE = 0.005
ITERATION_BATCH_SIZE = 300_000


class BaseSVDModelParams(AbstractSVDModelParams):

    def __init__(self):
        # non update self
        self.mean_rating: float = None

        # update self
        self.users_bias: np.ndarray = None
        self.items_bias: np.ndarray = None
        self.users_latent_features: np.ndarray = None
        self.items_latent_features: np.ndarray = None

    def initialize_parameters(self, data: pd.DataFrame, latent_dim: int):
        ratings = data.iloc[:, 2]
        self.mean_rating = ratings.sum() / len(ratings)

        users = data.iloc[:, 0]
        user_non_zero_count = np.bincount(users)
        user_non_zero_sum = np.bincount(users, weights=ratings)
        self.users_bias = (user_non_zero_sum / user_non_zero_count) - self.mean_rating

        items = data.iloc[:, 1]
        item_non_zero_count = np.bincount(items)
        item_non_zero_sum = np.bincount(items, weights=ratings)
        self.items_bias = (item_non_zero_sum / item_non_zero_count) - self.mean_rating

        scale = INITIALIZE_LATENT_FEATURES_SCALE
        n_users = users.max() + 1
        n_items = items.max() + 1

        self.users_latent_features = np.random.normal(0, scale, n_users * latent_dim) \
            .reshape(n_users, latent_dim)

        self.items_latent_features = np.random.normal(0, scale, n_items * latent_dim) \
            .reshape(n_items, latent_dim)

    def estimate_rating(self, user: int, item: int) -> float:
        mean_rating = self.mean_rating
        user_bias = self.users_bias[user]
        item_bias = self.items_bias[item]
        user_latent = self.users_latent_features[user]
        item_latent = self.items_latent_features[item]
        latent_product = np.dot(item_latent, user_latent)
        return mean_rating + user_bias + item_bias + latent_product

    def update(self, user: int, item: int, err: float, regularization: float, learning_rate: float):
        self.users_bias[user] += learning_rate * (err - (regularization * self.users_bias[user]))
        self.items_bias[item] += learning_rate * (err - (regularization * self.items_bias[item]))

        user_latent_features = self.users_latent_features[user]  # p_i
        item_latent_features = self.items_latent_features[item]  # q_i
        self.users_latent_features[user] += learning_rate * (
                    (err * item_latent_features) - (regularization * user_latent_features))
        self.items_latent_features[item] += learning_rate * (
                    (err * user_latent_features) - (regularization * item_latent_features))


def save_base_svd_model_parameters(svd_params: BaseSVDModelParams, filepath: str):
    np.savez_compressed(filepath,
                        mean_rating=svd_params.mean_rating,
                        users_bias=svd_params.users_bias,
                        items_bias=svd_params.items_bias,
                        users_latent=svd_params.users_latent_features,
                        items_latent=svd_params.items_latent_features
                        )


def load_svd_model(filepath: str) -> BaseSVDModelParams:
    svd_params = BaseSVDModelParams()
    loaded_params = np.load(filepath)
    svd_params.mean_rating = loaded_params["mean_rating"]
    svd_params.users_bias = loaded_params["users_bias"]
    svd_params.items_bias = loaded_params["items_bias"]
    svd_params.users_latent_features = loaded_params["users_latent"]
    svd_params.items_latent_features = loaded_params["items_latent"]
    return svd_params

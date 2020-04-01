import time
from math import sqrt
from random import shuffle, Random, sample
from typing import Sequence, NamedTuple, List, Tuple

import numpy as np
import scipy
import tqdm as tqdm
from scipy.sparse import spmatrix, find

from recsys.eval.evaltools import rmse

INITIALIZE_LATENT_FEATURES_SCALE = 0.005


class BaseModel(object):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """
    def __init__(self, learning_rate: float = 0.005, lr_decrease_factor: float = 0.99, regularization: float = 0.02, converge_threshold: float = 1e-4,
                 max_iterations: int = 30, score_sample_size: float = 100_000, random_seed: int = None):
        self.n_users: int = None
        self.n_items: int = None
        self.n_latent: int = None

        self.model_parameters_: SVDModelParams = None
        self.initial_learning_rate = learning_rate
        self.__adaptive_learning_rate = learning_rate
        self.lr_decrease_factor = lr_decrease_factor
        self.regularization = regularization
        self.converge_threshold = converge_threshold
        self.__converged = False
        self.score_sample_size = score_sample_size
        self.max_iterations = max_iterations

        self.random = Random(random_seed)

    def fit(self, data: spmatrix, n_latent: int):
        self.n_users, self.n_items = data.shape
        self.n_latent = n_latent
        print("initializing model parameters")
        self.model_parameters_ = initialize_parameters(data, n_latent)

        user, item, rating = scipy.sparse.find(data)
        ratings = list(zip(user, item, rating))
        shuffle(ratings, self.random.random)

        prev_score = self.__get_score(ratings)
        print(f"initial score: {prev_score}")
        num_iterations = 0
        while (not self.__converged) and (num_iterations < self.max_iterations):
            print(f"start iteration {num_iterations}")
            start = time.time()
            for (u, i, r) in tqdm.tqdm(ratings, total=len(ratings)):
                r_est = estimate_rating(u, i, self.model_parameters_)
                err = r - r_est
                self.model_parameters_.update(u, i, err, self.regularization, self.__adaptive_learning_rate)

            end = time.time()
            print(f"iteration {num_iterations} took {int(end - start)} sec")

            # check for convergence
            new_score = self.__get_score(ratings)
            print(f"new score: {new_score}")
            self.__converged = self.__is_converged(prev_score, new_score)
            prev_score = new_score

            # update values for the next iteration
            self.__adaptive_learning_rate *= self.lr_decrease_factor
            num_iterations += 1
            shuffle(ratings, self.random.random)

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass

    def __get_score(self, ratings) -> float:
        ratings_sample = sample(ratings, self.score_sample_size)
        r_true, r_pred = zip(*[(r, estimate_rating(u, i, self.model_parameters_)) for u, i, r in ratings_sample])
        return rmse(r_true, r_pred)

    def __is_converged(self, prev_score: float, new_score: float) -> bool:
        score_diff = prev_score - new_score
        if score_diff <= 0:
            return True

        return abs(score_diff) <= self.converge_threshold


class SVDModelParams(NamedTuple):
    mean_rating: float
    users_bias: np.ndarray
    items_bias: np.ndarray
    users_latent_features: np.ndarray
    items_latent_features: np.ndarray

    def update(self, user: int, item: int, err: float, regularization: float, learning_rate: float):
        self.users_bias[user] += learning_rate * (err - (regularization * self.users_bias[user]))
        self.items_bias[item] += learning_rate * (err - (regularization * self.items_bias[item]))

        user_latent_features = self.users_latent_features[user]
        item_latent_features = self.items_latent_features[item]
        self.users_latent_features[user] += learning_rate * ((err * item_latent_features) - (regularization * user_latent_features))
        self.items_latent_features[item] += learning_rate * ((err * user_latent_features) - (regularization * item_latent_features))


def initialize_parameters(users_items_matrix: spmatrix, latent_dim: int) -> SVDModelParams:
    user, item, rating = find(users_items_matrix)

    mean_rating = users_items_matrix.sum() / len(rating)

    user_non_zero_count = np.bincount(user)
    user_non_zero_sum = np.bincount(user, weights=rating)
    users_mean_rating = (user_non_zero_sum / user_non_zero_count) - mean_rating

    item_non_zero_count = np.bincount(item)
    item_non_zero_sum = np.bincount(item, weights=rating)
    items_mean_rating = (item_non_zero_sum / item_non_zero_count) - mean_rating

    scale = INITIALIZE_LATENT_FEATURES_SCALE
    n_users, n_items = users_items_matrix.shape
    latent_user_features = np.random.normal(0, scale, n_users * latent_dim)\
        .reshape(n_users, latent_dim)
    latent_item_features = np.random.normal(0, scale, n_items * latent_dim)\
        .reshape(n_items, latent_dim)

    return SVDModelParams(mean_rating,
                          users_mean_rating,
                          items_mean_rating,
                          latent_user_features,
                          latent_item_features
                          )


def estimate_rating(user: int, item: int, params: SVDModelParams) -> float:
    mean_rating = params.mean_rating
    user_bias = params.users_bias[user]
    item_bias = params.items_bias[item]
    user_latent = params.users_latent_features[user]
    item_latent = params.items_latent_features[item]


    latent_product = np.dot(item_latent, user_latent)
    return mean_rating + user_bias + item_bias + latent_product




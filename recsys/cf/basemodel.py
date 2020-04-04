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

INITIALIZE_LATENT_FEATURES_SCALE = 0.005
ITERATION_BATCH_SIZE = 300_000


class BaseModel(object):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    VALIDATION_SPLIT_RANDOM_SEED = 0.4
    VALIDATION_USERS_SIZE = 0.5
    VALIDATION_ITEMS_PER_USER_SIZE = 0.3
    

    def __init__(self, learning_rate: float = 0.02, lr_decrease_factor: float = 0.99, regularization: float = 0.02, converge_threshold: float = 1e-4,
                 max_iterations: int = 30, random_seed: int = None):
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
        self.max_iterations = max_iterations

        self.random = Random(random_seed)

    def fit(self, train_data: pd.DataFrame, n_latent: int):
        print("start base model")
        print("split train data for validation")
        train_data, validation_data = split_dataset(
            train_data, \
            self.VALIDATION_USERS_SIZE, \
            self.VALIDATION_ITEMS_PER_USER_SIZE \
        )
        validation_ratings = list(validation_data.itertuples(index=False, name=None))   # [(user, item, rating)....]

        self.n_users, self.n_items = train_data.shape
        self.n_latent = n_latent
        print("initializing model parameters")
        self.model_parameters_ = initialize_parameters(train_data, n_latent)

        prev_score = self.__get_score(validation_ratings)
        print(f"initial score: {prev_score}")
        num_iterations = 0
        while (not self.__converged) and (num_iterations < self.max_iterations):
            
            print("shuffle batch")
            train_ratings = train_data.sample(frac=1, random_state=self.random.randint(0, 100)).itertuples(index=False, name=None)
            train_ratings = islice(train_ratings, 300_000)
            
            print(f"start iteration {num_iterations}")
            start = time.time()
            for u, i, r in tqdm.tqdm(train_ratings, total=ITERATION_BATCH_SIZE):
                r_est = estimate_rating(u, i, self.model_parameters_)
                err = r - r_est
                self.model_parameters_.update(u, i, err, self.regularization, self.__adaptive_learning_rate)

            end = time.time()
            print(f"iteration {num_iterations} took {int(end - start)} sec")

            # check for convergence
            new_score = self.__get_score(validation_ratings)
            print(f"new score: {new_score}")
            self.__converged = self.__is_converged(prev_score, new_score)
            prev_score = new_score

            # update values for the next iteration
            self.__adaptive_learning_rate *= self.lr_decrease_factor
            num_iterations += 1

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass

    def __get_score(self, ratings) -> float:
        r_true, r_pred = zip(*[(r, estimate_rating(u, i, self.model_parameters_)) for u, i, r in ratings])
        return rmse(r_true, r_pred)

    def __is_converged(self, prev_score: float, new_score: float) -> bool:
        score_diff = prev_score - new_score
        # if score_diff <= 0:
        #     return True

        return abs(score_diff) <= self.converge_threshold


class SVDModelParams(NamedTuple):
    # non update params
    mean_rating: float

    # update params
    users_bias: np.ndarray
    items_bias: np.ndarray
    users_latent_features: np.ndarray
    items_latent_features: np.ndarray

    def update(self, user: int, item: int, err: float, regularization: float, learning_rate: float):
        self.users_bias[user] += learning_rate * (err - (regularization * self.users_bias[user]))
        self.items_bias[item] += learning_rate * (err - (regularization * self.items_bias[item]))

        user_latent_features = self.users_latent_features[user] # p_i
        item_latent_features = self.items_latent_features[item] # q_i
        self.users_latent_features[user] += learning_rate * ((err * item_latent_features) - (regularization * user_latent_features))
        self.items_latent_features[item] += learning_rate * ((err * user_latent_features) - (regularization * item_latent_features))


def initialize_parameters(data: pd.DataFrame, latent_dim: int) -> SVDModelParams:
    ratings = data.iloc[:, 2]
    mean_rating = ratings.sum() / len(ratings)

    users = data.iloc[:, 0]
    user_non_zero_count = np.bincount(users)
    user_non_zero_sum = np.bincount(users, weights=ratings)
    users_mean_rating = (user_non_zero_sum / user_non_zero_count) - mean_rating

    items = data.iloc[:, 1]
    item_non_zero_count = np.bincount(items)
    item_non_zero_sum = np.bincount(items, weights=ratings)
    items_mean_rating = (item_non_zero_sum / item_non_zero_count) - mean_rating

    scale = INITIALIZE_LATENT_FEATURES_SCALE
    n_users = users.max() + 1
    n_items = items.max() + 1
    
    latent_user_features = np.random.normal(0, scale, n_users * latent_dim)\
        .reshape(n_users, latent_dim)
    
    latent_item_features = np.random.normal(0, scale, n_items * latent_dim)\
        .reshape(n_items, latent_dim)

    return SVDModelParams(mean_rating,
                          users_mean_rating,
                          items_mean_rating,
                          latent_user_features,
                          latent_item_features,
                          )

def estimate_rating(user: int, item: int, params: SVDModelParams) -> float:
    mean_rating = params.mean_rating
    user_bias = params.users_bias[user]
    item_bias = params.items_bias[item]
    user_latent = params.users_latent_features[user]
    item_latent = params.items_latent_features[item]
    latent_product = np.dot(item_latent, user_latent)
    return mean_rating + user_bias + item_bias + latent_product

def save_svd_model(svd_model: BaseModel, filename: str):
    pass

def load_svd_model(filename: str) -> BaseModel:
    pass

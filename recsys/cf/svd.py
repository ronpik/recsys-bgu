import abc

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
from recsys.cf import AbstractSVDModelParams

INITIALIZE_LATENT_FEATURES_SCALE = 0.005
ITERATION_BATCH_SIZE = 300_000


class SVDModelEngine(abc.ABC):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    VALIDATION_SPLIT_RANDOM_SEED = 0.4
    VALIDATION_USERS_SIZE = 0.5
    VALIDATION_ITEMS_PER_USER_SIZE = 0.3

    def __init__(self, svd_parameters: AbstractSVDModelParams, learning_rate: float = 0.05, lr_decrease_factor: float = 0.99, regularization: float = 0.02, converge_threshold: float = 1e-4,
                 max_iterations: int = 0, random_seed: int = None):
        self.n_users: int = None
        self.n_items: int = None
        self.n_latent: int = None

        self.__model_parameters = svd_parameters

        self.initial_learning_rate = learning_rate
        self.__adaptive_learning_rate = learning_rate
        self.lr_decrease_factor = lr_decrease_factor
        self.regularization = regularization
        self.converge_threshold = converge_threshold
        self.__converged = False
        self.max_iterations = max_iterations

        self.random = Random(random_seed)
    
    @property
    def model_parameters_(self) -> AbstractSVDModelParams:
        return self.__model_parameters

    def fit(self, train_data: pd.DataFrame, n_latent: int):
        """
        train svd model (matrix factorization process with SGD) with latent features of dimension [n_latent]
        """
        self.__initialize_model_parameters(train_data, n_latent)

        print("split train data for validation")
        start = time.time()
        train_data, validation_data = split_dataset(
            train_data, self.VALIDATION_USERS_SIZE, self.VALIDATION_ITEMS_PER_USER_SIZE)
        end = time.time()
        print(f"split to train - validation datasets took {end - start: .2f} sec")

        validation_ratings = list(
            validation_data.itertuples(index=False, name=None))
        prev_score = self.__get_score(validation_ratings)
        print(f"initial score: {prev_score}")

        num_iterations = 0
        while (not self.__converged) and (num_iterations < self.max_iterations):
            print("shuffle batch")
            train_ratings = train_data \
                .sample(frac=1, random_state=self.random.randint(0, 100)) \
                .itertuples(index=False, name=None)
            train_ratings = islice(train_ratings, 300_000)

            print(f"start iteration {num_iterations}")
            start = time.time()
            for u, i, r in tqdm.tqdm(train_ratings, total=ITERATION_BATCH_SIZE):
                r_est = self.model_parameters_.estimate_rating(u, i)
                err = r - r_est
                self.model_parameters_.update(
                    u, i, err, self.regularization, self.__adaptive_learning_rate)

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

    def __initialize_model_parameters(self, train_data: pd.DataFrame, n_latent: int):
        self.n_users, self.n_items = train_data.shape
        self.n_latent = n_latent
        print("initializing model parameters")
        self.model_parameters_.initialize_parameters(train_data, n_latent)

    def predict(self, data: np.ndarray) -> Sequence[float]:
        pass

    def __get_score(self, ratings) -> float:
        r_true, r_pred = zip(*[(r, self.model_parameters_.estimate_rating(u, i)) \
                                for u, i, r in ratings])
        return rmse(r_true, r_pred)

    def __is_converged(self, prev_score: float, new_score: float) -> bool:
        score_diff = prev_score - new_score
        # if score_diff <= 0:
        #     return True

        return abs(score_diff) <= self.converge_threshold

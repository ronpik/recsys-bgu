
import abc

import time
from copy import deepcopy
from math import sqrt
from random import shuffle, Random, sample
from typing import Sequence, NamedTuple, List, Tuple
from itertools import islice, starmap

import pandas as pd
import numpy as np
import tqdm as tqdm

from recsys.eval.evaltools import rmse
from recsys.utils.data.yelp_dataset import split_dataset
from recsys.cf import AbstractSVDModelParams

INITIALIZE_LATENT_FEATURES_SCALE = 0.01
ITERATION_BATCH_SIZE = 100_000


class SVDModelEngine(abc.ABC):
    """
    Implementing SVD for recommendation systems (i.e the given data is sparse due to large amount of missing values)
    """

    VALIDATION_USERS_SIZE = 0.5
    VALIDATION_ITEMS_PER_USER_SIZE = 0.4

    def __init__(self, svd_parameters: AbstractSVDModelParams,
                 learning_rate: float = 0.01,
                 lr_decrease_factor: float = 0.9,
                 regularization: float = 0.1,
                 converge_threshold: float = 1e-5,
                 max_iterations: int = 30,
                 random_seed: int = None
                 ):
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
        print(f"new train-set has {len(train_data)} records")
        print(f"validation-set has {len(validation_data)} records")

        validation_ratings = list(
            validation_data.itertuples(index=False, name=None))
        prev_score = self.__get_score(validation_ratings)
        print(f"initial score: {prev_score}")

        best_params = deepcopy(self.model_parameters_)

        num_iterations = 1
        while (not self.__converged) and (num_iterations < self.max_iterations):
            print("shuffle batch")
            train_ratings = train_data \
                .sample(frac=1, random_state=self.random.randint(0, 100)) \
                .itertuples(index=False, name=None)

            batch_size = min(ITERATION_BATCH_SIZE, len(train_data))
            num_batches = len(train_data) // ITERATION_BATCH_SIZE + 1

            print(f"\nstart iteration {num_iterations}")
            iteration_start = time.time()
            for batch_num in range(1, num_batches + 1):
                batch = islice(train_ratings, batch_size)
                print(f"start batch {batch_num}")
                for u, i, r in tqdm.tqdm(batch, total=batch_size):
                    r_est = self.model_parameters_.estimate_rating(u, i)
                    err = r - r_est
                    self.model_parameters_.update(
                        u, i, err, self.regularization, self.__adaptive_learning_rate)

                score_after_batch = self.__get_score(validation_ratings)
                print(f"intermediate validation score: {score_after_batch}")

            iteration_end = time.time()
            print(f"iteration {num_iterations} took {int(iteration_end - iteration_start)} sec")

            # check for convergence
            print("calculate train score")
            train_score = self.__get_score(list(train_data.itertuples(index=False, name=None)))
            print(f"train score: {train_score}")
            new_score = self.__get_score(validation_ratings)
            print(f"validation score: {new_score}")
            self.__converged = self.__is_converged(prev_score, new_score)
            if new_score < prev_score:
                best_params = deepcopy(self.model_parameters_)

            prev_score = new_score

            # update values for the next iteration
            self.__adaptive_learning_rate *= self.lr_decrease_factor
            num_iterations += 1

        self.__model_parameters = best_params

    def __initialize_model_parameters(self, train_data: pd.DataFrame, n_latent: int):
        self.n_users, self.n_items = train_data.shape
        self.n_latent = n_latent
        print("initializing model parameters")
        self.model_parameters_.initialize_parameters(train_data, n_latent)

    def predict(self, users: Sequence[int], items: Sequence[int]) -> Sequence[float]:
        user_item_pairs = zip(users, items)
        pred_single = self.model_parameters_.estimate_rating
        return np.asarray(list(starmap(pred_single, user_item_pairs)))

    def __get_score(self, ratings) -> float:
        r_true, r_pred = zip(*[(r, self.model_parameters_.estimate_rating(u, i)) \
                                for u, i, r in ratings])
        return rmse(r_true, r_pred)

    def __is_converged(self, prev_score: float, new_score: float) -> bool:
        score_diff = prev_score - new_score
        if score_diff <= 0:
            return True

        return abs(score_diff) <= self.converge_threshold

import time
from typing import Sequence, Any

import pandas as pd

from recsys.cf import SVDModelEngine
from recsys.cf.basemodel import BaseSVDModelParams
from recsys.cf.advancedmodel import AdvancedSVDModelParams
from recsys.cf.combined import CombinedModel
from recsys.cf.sentiment import SentimentModel
from recsys.cf.surprise import SupriseModel
from recsys.utils.data.yelp_dataset import prepare_data_for_cf, split_dataset


class RecommenderSystem(object):

    VALIDATION_SPLIT_RANDOM_SEED = 0.4
    VALIDATION_USERS_SIZE = 0.5
    VALIDATION_ITEMS_PER_USER_SIZE = 0.3

    def __init__(self, random_seed: int = None):
        self.base_model = None
        self.advanced_model = None
        self.surprise_model = None
        self.sentiment_model = None
        self.combined_model = None
        self.random_seed = random_seed

    def TrainBaseModel(self, train_data: pd.DataFrame, n_latent: int):
        print(f"number of latent features: {n_latent}")
        base_svd_parameters = BaseSVDModelParams()
        self.base_model = SVDModelEngine(base_svd_parameters, random_seed=self.random_seed)
        self.base_model.fit(train_data, n_latent)

    def TrainAdvancedModel(self, train_data: pd.DataFrame, n_latent: int):
        print(f"number of latent features: {n_latent}")
        advanced_svd_parameters = AdvancedSVDModelParams()
        self.advanced_model = SVDModelEngine(advanced_svd_parameters, random_seed=self.random_seed)
        self.advanced_model.fit(train_data, n_latent)

    def TrainSentimentModel(self, X_train: pd.DataFrame, y_train: Sequence[int]):
        self.sentiment_model = SentimentModel()
        self.sentiment_model.fit(X_train, y_train)

    def TrainSupriseModel(self, train_data):
        self.surprise_model = SupriseModel()
        self.surprise_model.fit(train_data)

    def TrainCombinedModel(self):
        """
        Bonus - only if there is time
        :return:
        """
        # TODO(train the model)
        self.base_model = CombinedModel()

    def PredictRating(self, data: pd.DataFrame, model_name="svd") -> Sequence[int]:
        if model_name.startswith("svd"):
            users = data.user_id
            items = data.business_id

            if model_name == "svd++":
                return self.advanced_model.predict(users, items)
            elif model_name == "svd":
                return self.base_model.predict(users, items)

        elif model_name == "sentiment":
            return self.sentiment_model.predict(data)
        elif model_name == "combined":
            return self.combined_model.predict(data)
        elif model_name == "surprise":
            return self.surprise_model.predict(data)

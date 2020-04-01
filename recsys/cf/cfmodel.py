import pandas as pd

from recsys.cf.basemodel import BaseModel
from recsys.cf.advancedmodel import AdvancedModel
from recsys.cf.combined import CombinedModel
from recsys.cf.sentiment import SentimentModel
from recsys.utils.data.yelp_dataset import prepare_data_for_cf, split_dataset
import time


class RecommenderSystem(object):

    VALIDATION_SPLIT_RANDOM_SEED = 0.4
    VALIDATION_USERS_SIZE = 0.5
    VALIDATION_ITEMS_PER_USER_SIZE = 0.3

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame): 
        self.train_data, self.validation_data = split_dataset( \
            train_data, \
            self.VALIDATION_USERS_SIZE, \
            self.VALIDATION_ITEMS_PER_USER_SIZE \
        )
        self.test_data = test_data

        self.base_model = None
        self.sentiment_model = None
        self.combined_model = None

    def TrainBaseModel(self, n_latent: int):
        print(f"number of latent features: {n_latent}")
        train_mat, validation_mat = prepare_data_for_cf(self.train_data, self.validation_data)
        self.base_model = BaseModel()
        self.base_model.fit(train_mat, validation_mat, n_latent)

    def TrainAdvancedModel(self, n_latent: int):
        print(f"number of latent features: {n_latent}")
        train_mat, validation_mat = prepare_data_for_cf(self.train_data, self.validation_data)
        self.advanced_model = AdvancedModel()
        self.advanced_model.fit(train_mat, validation_mat, n_latent)

    def TrainSentimentModel(self):
        self.sentiment_model = SentimentModel()
        self.sentiment_model.fit(self.train_data.drop('stars'), self.train_data.stars)

    def TrainCombinedModel(self):
        """
        Bonus - only if there is time
        :return:
        """
        # TODO(train the model)
        self.base_model = CombinedModel()

    def PredictRating(self, data, model_name="base"):
        if model_name == "base":
            return self.base_model.predict(data)
        if model_name == "sentiment":
            return self.sentiment_model.predict(data)
        if model_name == "combined":
            return self.combined_model.predict(data)

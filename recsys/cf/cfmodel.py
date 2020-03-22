import pandas as pd
from scipy.sparse import spmatrix

from recsys.cf.basemodel import BaseModel
from recsys.cf.combined import CombinedModel
from recsys.cf.sentiment import SentimentModel


class RecommenderSystem(object):
    def __init__(self):
        self.base_model = None
        self.sentiment_model = None
        self.combined_model = None

    def TrainBaseModel(self, data: spmatrix, p: int, q: int, learning_rate: float = 0.4, regularization: float = 10):
        self.base_model = BaseModel()
        self.base_model.fit(data, p, q)

    def TrainSentimentModel(self):
        # TODO(train the model)
        self.sentiment_model = SentimentModel()

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

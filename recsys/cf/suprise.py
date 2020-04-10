from surprise import SVDpp
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
import pandas as pd
import tqdm

class SupriseModel(object):
    def __init__(self):
        self.model = None
        self.reader = None
        self.train_surprise = None
        self.trainset = None

    def fit(self, data: pd.DataFrame):
        self.reader = Reader(rating_scale=(1, 5))
        self.train_surprise = Dataset.load_from_df(data, self.reader)
        self.trainset = self.train_surprise.build_full_trainset()
        self.model = SVDpp(lr_all = 0.002, reg_all = 0.06, verbose = True,n_epochs=10)
        self.model.fit(self.trainset)
        
    def predict(self, data) -> pd.Series:
        predictions = []
        for _, row in data.iterrows():
            pred = self.model.predict(row['user_id'], row['business_id']).est
            predictions.append(pred)
        
        return pd.Series(predictions)

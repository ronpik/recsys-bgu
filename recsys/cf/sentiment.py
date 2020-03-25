from typing import Sequence
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tqdm
from sklearn import linear_model

class SentimentModel(object):
    def __init__(self):
        self.model = None
        

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self.model = linear_model.LinearRegression()
        self.model.fit(data, y)
        

    def predict(self, data) -> pd.Series:
        return self.model.predict(data)
        

def sentiment_scores(text:str) -> dict:
    analyser = SentimentIntensityAnalyzer()
    return analyser.polarity_scores(text)


def count_symbol(text:str, symbol:str) -> int:
    text = str(text)
    return {f'count_{symbol}':text.count(symbol)}


def text_length(text:str) -> int:
    return {'text_length': len(text)}


def generate_feature_set(text: pd.Series) -> pd.DataFrame:
    
    feature_set = []
    for index, txt in tqdm.tqdm(text.items()):
        scores = sentiment_scores(txt)
        scores.update({'index': index})
        scores.update(count_symbol(txt, '!'))
        scores.update(count_symbol(txt, '?'))
        scores.update(text_length(txt))
        feature_set.append(scores)

    return pd.DataFrame(feature_set)





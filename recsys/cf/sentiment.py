from typing import Sequence
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentModel(object):
    def __init__(self):
        pass

    def fit(self, data, y: Sequence[float]):
        pass

    def predict(self, data) -> Sequence[float]:
        pass



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
    for index, txt in text.items():
        scores = sentiment_scores(txt)
        scores.update({'index': index})
        scores.update(count_symbol(txt, '!'))
        scores.update(count_symbol(txt, '?'))
        scores.update(text_length(txt))
        feature_set.append(scores)

    return pd.DataFrame(feature_set)

s = pd.Series(['hi love you!!', 'stupid shit', 'it is ok???'])

x = generate_feature_set(s)




import time
from math import sqrt
from random import shuffle, Random, sample
from typing import Sequence, NamedTuple, List, Tuple

import numpy as np
import pandas as pd
import tqdm as tqdm
from itertools import groupby
from operator import itemgetter

from recsys.eval.evaltools import rmse
from recsys.cf import AbstractSVDModel
from recsys.cf.basemodel import BaseSVDModelParams

INITIALIZE_LATENT_FEATURES_SCALE = 0.005


class AdvancedSVDModelParams(BaseSVDModelParams):
    
    def __init__(self):
        super().__init__()
        self.user_items_mapping: list = None
        self.itemspp: np.ndarray = None
    
    def initialize_parameters(self, data: pd.DataFrame, latent_dim: int):
        super().initialize_parameters(data, latent_dim)

        self.user_items_mapping = create_user_item_mapping(data)

        scale = INITIALIZE_LATENT_FEATURES_SCALE
        n_items = len(self.items_bias)
        self.itemspp = np.random.normal(0, scale, n_items * latent_dim) \
            .reshape(n_items, latent_dim)

    def update(self, user: int, item: int, err: float, regularization: float, learning_rate: float):
        super().update()
        user_items_set = self.user_items_mapping[user]
        norm_factor = np.sqrt(len(user_items_set))
        for i in user_items_set:
            self.itemspp[i] += learning_rate * ((err * self.items_latent_features) / norm_factor) - (regularization * self.itemspp[i])

    def estimate_rating(self, user: int, item: int) -> float:
        user_items_mask = self.user_items_mapping[user]
        user_itemspp = np.sum(self.itemspp[user_items_mask], axis=0)   # sum over the rows
        user_itemspp /= len(np.sqrt(user_items_mask))
        item_latent = self.items_latent_features[item]
        user_addition_latent_product = np.dot(item_latent, user_itemspp)
        
        base_estimate = super().estimate_rating(user, item)
        return base_estimate + user_addition_latent_product


def create_user_item_mapping(train_data: pd.DataFrame) -> list:
    user_items_mapping = {user_id: np.array(list(group.business_id)) for user_id, group in train_data.groupby('user_id')}
    return [user_items_mapping[i] for i in range(len(user_items_mapping))]

    

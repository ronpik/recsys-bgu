import time
from typing import List

import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter

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
        super().update(user, item, err, regularization, learning_rate)
        items_mask = self.user_items_mapping[user]
        norm_factor = np.sqrt(len(items_mask))
        user_items_update = ((err * self.items_latent_features[items_mask]) / norm_factor) - (regularization * self.itemspp[items_mask])
        self.itemspp[items_mask] += learning_rate * user_items_update

    def estimate_rating(self, user: int, item: int) -> float:
        user_items_mask = self.user_items_mapping[user]
        user_itemspp = np.sum(self.itemspp[user_items_mask], axis=0)   # sum over the rows
        user_itemspp /= len(np.sqrt(user_items_mask))
        item_latent = self.items_latent_features[item]
        user_addition_latent_product = np.dot(item_latent, user_itemspp)
        
        base_estimate = super().estimate_rating(user, item)
        return base_estimate + user_addition_latent_product


def create_user_item_mapping(train_data: pd.DataFrame) -> list:
    print("create mapping between users to the items they rated")
    start = time.time()
    user_item_ratings = sorted(train_data.itertuples(index=False, name=None), key=itemgetter(0))
    grouped_items_by_user = groupby(user_item_ratings, key=itemgetter(0))
    rated_items_by_user_index = map(itemgetter(1), grouped_items_by_user)
    item_ids_by_user_index = map(lambda group: list(map(itemgetter(1), group)), rated_items_by_user_index)
    user_items_mapping = list(map(np.array, item_ids_by_user_index))

    end = time.time()
    print(f"creating user items mapping took {end - start:.2f} sec")
    return user_items_mapping


def encode_user_items_mapping(user_item_mapping: List[np.ndarray]) -> np.array:
    total_size = sum(len(items) for items in user_item_mapping) + len(user_item_mapping)
    encoded = np.zeros(total_size, dtype=int)

    start = 0
    for items in user_item_mapping:
        num_items = len(items)
        encoded[start] = num_items
        encoded[start + 1: start + 1 + num_items] = items
        start += 1 + num_items

    assert(start == total_size)
    return encoded


def decode_user_items_mapping(encoded_mapping: np.array) -> List[np.array]:
    user_items_mapping = []
    start = 0
    while start < len(encoded_mapping):
        num_items = encoded_mapping[start]
        items = encoded_mapping[start + 1: start + 1 + num_items]
        user_items_mapping.append(items)

        start += 1 + num_items

    return user_items_mapping


def save_advanced_svd_model(svd_params: AdvancedSVDModelParams, filepath: str):
    np.savez_compressed(filepath,
                        mean_rating=svd_params.mean_rating,
                        users_bias=svd_params.users_bias,
                        items_bias=svd_params.items_bias,
                        users_latent=svd_params.users_latent_features,
                        items_latent=svd_params.items_latent_features,
                        user_items_mapping=encode_user_items_mapping(svd_params.user_items_mapping),
                        itemspp=svd_params.itemspp
                        )


def load_advanced_svd_model(filepath: str) -> AdvancedSVDModelParams:
    svd_params = AdvancedSVDModelParams()
    loaded_params = np.load(filepath)
    svd_params.mean_rating = loaded_params["mean_rating"]
    svd_params.users_bias = loaded_params["users_bias"]
    svd_params.items_bias = loaded_params["items_bias"]
    svd_params.users_latent_features = loaded_params["users_latent"]
    svd_params.items_latent_features = loaded_params["items_latent"]
    svd_params.user_items_mapping = decode_user_items_mapping(loaded_params["user_items_mapping"])
    svd_params.itemspp = loaded_params["itemspp"]
    return svd_params

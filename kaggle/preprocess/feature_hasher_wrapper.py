from typing import Sequence, Tuple, Dict, List

import numpy as np
import scipy.sparse
from sklearn.feature_extraction import FeatureHasher

from kaggle.preprocess.utils import filter_by_occurrence, create_categories_index_mapping


class FeatureFilterHasher(object):
    def __init__(self, feature_prefix: str, min_occurrence_threshold: int = 0, **hasher_kwargs):
        self.feature_prefix = feature_prefix
        self.min_occurrence_threshold = min_occurrence_threshold
        self.hasher_kwargs = hasher_kwargs
        self.hasher_kwargs["input_type"] = "string"

        # post fit members
        self.num_features: int = None
        self.hasher: FeatureHasher = None
        self.features_names: Sequence[str] = None

    def fit(self, values: Sequence[str]) -> 'FeatureFilterHasher':
        if (self.min_occurrence_threshold > 0):
            values = filter_by_occurrence(values, self.min_occurrence_threshold)

        self.num_features = derive_num_features(len(values))
        self.hasher = FeatureHasher(n_features=self.num_features, **self.hasher_kwargs)
        self.features_names = [f"{self.feature_prefix}:{i}" for i in range(self.num_features)]
        return self

    def transform(self, values: Sequence[str]) -> scipy.sparse:
        return self.hasher.transform(values)


def derive_num_features(num_categories: int) -> int:
    return int(np.sqrt(num_categories))
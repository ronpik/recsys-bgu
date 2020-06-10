from itertools import chain
from operator import itemgetter
from typing import Tuple, Sequence, Set, Any, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from kaggle.preprocess.feature_hasher_wrapper import FeatureFilterHasher
from kaggle.preprocess.numeric import get_numeric_headers, process_numeric_features
from kaggle.preprocess.temporal import VIEW_TIME_FIELD, get_temporal_headers, process_temporal_features
from kaggle.preprocess.utils import filter_by_occurrence, create_categories_index_mapping, \
    NON_FREQ_NAME


# todo replace dict by list of tuples to maintain order in all python versions
FEATURES_MIN_THRESHOLDS = [
    ("target_id_hash", 5),
    ("campaign_id_hash", 5),
    ("syndicator_id_hash", 5),
    ("placement_id_hash", 5),
    ("source_id_hash", 5),
    ("country_code", 5),
    ("region", 5)
]

TEMPORAL_FEATURES = [
    "page_view_start_time",
    "day_of_week",
    "time_of_day",
    "gmt_offset"
]

OTHER_CATEGORICAL_FEATURES = [
    "publisher_id_hash",
    "source_item_type",
    "browser_platform",
]

TARGET_TAXONOMY_FIELD = "target_item_taxonomy"
NO_TAX_CATEGORY_VALUE = "NO_CATEGORY"

OS_FIELD = "os_family"
NUM_OS_TYPES = 7

LABEL = "is_click"


class FeaturesHashingProcessor(object):
    def __init__(self, fillna: bool = True):
        self.fillna = fillna

        # map between field and category name to the index in the one hot encoding og this feature
        self.page_view_min_time: int = None
        self.features_names: List[str] = None
        self.__taxonomy_max_depth: int = None
        self.hashers: Dict[str, FeatureFilterHasher] = None

    def fit(self, data: pd.DataFrame) -> 'FeaturesHashingProcessor':
        """

        :param data:
        :return:
        """
        if self.fillna:
            categorical_features = list(map(itemgetter(0), FEATURES_MIN_THRESHOLDS)) + OTHER_CATEGORICAL_FEATURES
            categorical_features_to_fill = {c: NON_FREQ_NAME for c in categorical_features}
            data.fillna(categorical_features_to_fill, inplace=True)

        self.hashers = {}
        headers = []
        for field, threshold in FEATURES_MIN_THRESHOLDS:
            hasher = FeatureFilterHasher(feature_prefix=field, min_occurrence_threshold=threshold)
            self.hashers[field] = hasher.fit(data[field])
            headers.append(hasher.features_names)

        for field in OTHER_CATEGORICAL_FEATURES:
            assert(field not in FEATURES_MIN_THRESHOLDS)    # just in case, to prevent duplicated huge one hot feature!
            hasher = FeatureFilterHasher(feature_prefix=field)
            self.hashers[field] = hasher.fit(data[field])
            headers.append(hasher.features_names)

        tax_hashers, tax_features_names = create_taxonomy_hashers(data[TARGET_TAXONOMY_FIELD])
        headers.append(tax_features_names)
        self.__taxonomy_max_depth = len(tax_hashers)
        for i in range(len(tax_hashers)):
            mapping_name = f"tax{i}"
            self.hashers[mapping_name] = tax_hashers[i]

        headers.append(get_os_feature_names())

        numeric_headers = get_numeric_headers()
        headers.append(numeric_headers)

        self.page_view_min_time = min(data[VIEW_TIME_FIELD])
        temporal_headers = get_temporal_headers()
        headers.append(temporal_headers)

        self.features_names = list(chain(*headers))
        return self

    def transform(self, data: pd.DataFrame) -> csr_matrix:
        """

        :param data:
        :return:
        """
        features: List[scipy.sparse.spmatrix] = []
        categorical_features = list(map(itemgetter(0), FEATURES_MIN_THRESHOLDS)) + OTHER_CATEGORICAL_FEATURES
        for field in categorical_features:
            print(field)
            categories = np.asarray(data[field]).reshape(-1, 1)
            values = self.hashers[field].transform(categories)
            features.append(values)

        # taxonomy features
        hashers_by_depth = [self.hashers[f"tax{i}"] for i in range(self.__taxonomy_max_depth)]
        values = create_taxonomy_features(data[TARGET_TAXONOMY_FIELD], hashers_by_depth)
        features.append(values)

        os_features = encode_os_feature(data[OS_FIELD])
        features.append(os_features)

        numeric_features = process_numeric_features(data)
        features.append(numeric_features)

        temporal_features = process_temporal_features(data, self.page_view_min_time)
        features.append(temporal_features)

        return sparse_hstack(features, format="csr")

    def fit_transform(self, data: pd.DataFrame) -> csr_matrix:
        return self.fit(data)\
            .transform(data)


def get_categories_by_depth(taxonomy_values: Sequence[str]) -> List[List[str]]:
    """
    extract the taxonomy strings into smaller categories with hierarchy.
    :param taxonomy_values: list of taxonomy values, where each value contains the ancestors categories, ordered and splitted by '~'.
    :return: list of lists of categories. the category list at index i, corresponds to the categories in depth i in the taxonomy.
    """
    categories_by_hierarchy = [[], [], []]
    for tax in taxonomy_values:
        tax_categories = tax.split("~")
        for i in range(len(categories_by_hierarchy)):
            category = tax_categories[i] if i < len(tax_categories) else NO_TAX_CATEGORY_VALUE
            categories_by_hierarchy[i].append(category)

    return categories_by_hierarchy


def create_taxonomy_hashers(taxonomy_values: Sequence[str]) -> Tuple[List[FeatureFilterHasher], Sequence[str]]:
    categories_by_hierarchy = get_categories_by_depth(taxonomy_values)

    hashers = []
    taxonomy_features_names = []
    for i in range(len(categories_by_hierarchy)):
        values = categories_by_hierarchy[i]
        prefix = f"tax{i}"
        hasher = FeatureFilterHasher(feature_prefix=prefix)
        hashers.append(hasher.fit(values))
        taxonomy_features_names.extend(hasher.features_names)

    return hashers, taxonomy_features_names


def create_taxonomy_features(taxonomy_values: Sequence[str], hashers_by_depth: List[FeatureFilterHasher]) -> csr_matrix:
    categories_by_hierarchy = get_categories_by_depth(taxonomy_values)
    features_sets = []
    for i in range(len(categories_by_hierarchy)):
        categories = np.asarray(categories_by_hierarchy[i]).reshape(-1, 1)
        values = hashers_by_depth[i].transform(categories)
        features_sets.append(values)

    tax_features = sparse_hstack(features_sets, format="csr")
    return tax_features


def get_os_feature_names() -> Sequence[str]:
    return [f"os_{i}" for i in range(NUM_OS_TYPES)]


def encode_os_feature(values: Sequence[int]) -> np.ndarray:
    ohe = np.zeros((len(values), NUM_OS_TYPES), dtype=np.int8)
    for i, os_type in enumerate(values):
        ohe[i][os_type] = True

    return ohe

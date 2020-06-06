from itertools import chain
from operator import itemgetter
from typing import Tuple, Sequence, Set, Any, Dict, List

import numpy as np
import pandas as pd

from kaggle.preprocess.numeric import get_numeric_headers, process_numeric_features
from kaggle.preprocess.temporal import VIEW_TIME_FIELD, get_temporal_headers, process_temporal_features
from kaggle.preprocess.utils import filter_by_occurrence, to_one_hot_encoding, create_categories_index_mapping, \
    NON_FREQ_NAME

# todo replace dict by list of tuples to maintain order in all python versions
FEATURES_MIN_THRESHOLDS = [
    ("target_id_hash", 25),
    ("campaign_id_hash", 25),
    ("syndicator_id_hash", 25),
    ("placement_id_hash", 30),
    ("source_id_hash", 15),
    ("country_code", 30),
    ("region", 30)
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
    "os_family",
]

TARGET_TAXONOMY_FIELD = "target_item_taxonomy"
NO_TAX_CATEGORY_VALUE = "NO_CATEGORY"

LABEL = "is_click"


class FeaturesProcessor(object):
    def __init__(self, onehot: bool = False):
        self.__onehot = onehot

        # map between field and category name to the index in the one hot encoding og this feature
        self.ohe_index_mapping: Dict[str, Dict[str, int]] = None
        self.page_view_min_time: int = None
        self.features_names: List[str] = None
        self.__taxonomy_max_depth: int = None

    def fit(self, data: pd.DataFrame) -> 'FeaturesProcessor':
        """

        :param data:
        :return:
        """
        self.ohe_index_mapping = {}
        headers = []
        for field, threshold in FEATURES_MIN_THRESHOLDS:
            index_mapping, header = create_ohe_index_with_header(data[field], field, threshold)
            self.ohe_index_mapping[field] = index_mapping
            if self.__onehot:
                headers.append(header)
            else:
                headers.append([field])

        for field in OTHER_CATEGORICAL_FEATURES:
            assert(field not in FEATURES_MIN_THRESHOLDS)    # just in case, to prevent duplicated huge one hot feature!
            index_mapping, header = create_ohe_index_with_header(data[field], field)
            self.ohe_index_mapping[field] = index_mapping
            if self.__onehot:
                headers.append(header)
            else:
                headers.append([field])

        index_mappings_by_depth, headers_by_depth = create_taxonomy_index(data[TARGET_TAXONOMY_FIELD])
        self.__taxonomy_max_depth = len(index_mappings_by_depth)
        for i in range(len(index_mappings_by_depth)):
            mapping_name = f"tax{i}"
            tax_index_mapping = index_mappings_by_depth[i]
            self.ohe_index_mapping[mapping_name] = tax_index_mapping

        if self.__onehot:
            headers.append(list(chain(*headers_by_depth)))
        else:
            headers.append([f"tax{i}" for i in range(len(index_mappings_by_depth))])

        numeric_headers = get_numeric_headers()
        headers.append(numeric_headers)

        self.page_view_min_time = min(data[VIEW_TIME_FIELD])
        temporal_headers = get_temporal_headers()
        headers.append(temporal_headers)

        self.features_names = list(chain(*headers))
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """

        :param data:
        :return:
        """
        features = []
        categorical_features = list(map(itemgetter(0), FEATURES_MIN_THRESHOLDS)) + OTHER_CATEGORICAL_FEATURES
        for field in categorical_features:
            index_mapping = self.ohe_index_mapping[field]
            if self.__onehot:
                values = create_ohe_features(data[field], index_mapping)
            else:
                values = np.array([item if item in index_mapping else NON_FREQ_NAME for item in data[field]]).reshape(-1, 1)

            features.append(values)

        # taxonomy features
        index_mappings_by_depth = []
        for i in range(self.__taxonomy_max_depth):
            index_mapping = self.ohe_index_mapping[f"tax{i}"]
            index_mappings_by_depth.append(index_mapping)

        values = create_taxonomy_features(data[TARGET_TAXONOMY_FIELD], index_mappings_by_depth, self.__onehot)
        features.append(values)

        numeric_features = process_numeric_features(data)
        features.append(numeric_features)

        temporal_features = process_temporal_features(data, self.page_view_min_time)
        features.append(temporal_features)

        return np.hstack(features)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data)\
            .transform(data)


def create_ohe_index_with_header(values: Sequence[str],
                                 field_name: str,
                                 min_count: int = 0
) -> Tuple[Dict[str, int], List[str]]:
    if min_count > 0:
        values = filter_by_occurrence(values, min_count)

    index_mapping = create_categories_index_mapping(values)
    header = [f"{field_name}:{name}" for name in index_mapping]
    return index_mapping, header


def create_ohe_features(values: Sequence[str], categories_index_mapping: Dict[str, int]) -> np.ndarray:
    num_unique = len(categories_index_mapping)
    non_freq_item_index = categories_index_mapping.get(NON_FREQ_NAME)
    ohe = np.zeros((len(values), num_unique), dtype=bool)
    for i, item in enumerate(values):
        item_index = categories_index_mapping.get(item, non_freq_item_index)
        ohe[i][item_index] = True

    return ohe


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


def create_taxonomy_index(taxonomy_values: Sequence[str]) -> Tuple[List[Dict[str, int]], Sequence[str]]:
    categories_by_hierarchy = get_categories_by_depth(taxonomy_values)

    mappings = []
    full_header = []
    for i in range(len(categories_by_hierarchy)):
        values = categories_by_hierarchy[i]
        prefix = f"tax{i}"
        index_mapping, header = create_ohe_index_with_header(values, prefix)
        mappings.append(index_mapping)
        full_header.extend(header)

    return mappings, full_header


def create_taxonomy_features(taxonomy_values: Sequence[str], index_mappings_by_depth: List[Dict[str, int]], onehot: bool = False) -> np.ndarray:
    categories_by_hierarchy = get_categories_by_depth(taxonomy_values)

    ohe_sets = []
    for i in range(len(categories_by_hierarchy)):
        categories = categories_by_hierarchy[i]
        index_mapping = index_mappings_by_depth[i]
        if onehot:
            values = create_ohe_features(categories, index_mapping)
        else:
            values = np.array([item if item in index_mapping else NON_FREQ_NAME for item in categories]).reshape(-1, 1)

        ohe_sets.append(values)

    tax_features = np.hstack(ohe_sets)
    return tax_features

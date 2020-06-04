from itertools import chain
from typing import Tuple, Sequence, Set, Any, Dict

import numpy as np
import pandas as pd
import sklearn

from kaggle.preprocess.utils import filter_by_occurrence, to_one_hot_encoding

FEATURES_MIN_THRESHOLDS = {
    "target_id_hash": 25,
    "campaign_id_hash": 25,
    "syndicator_id_hash": 25,
    "placement_id_hash": 30,
    "source_id_hash": 15,
    "country_code": 30,
    "region": 30,
}

TIME_FEATURES = {
    "page_view_start_time",
    "day_of_week",
    "time_of_day",
    "gmt_offset"
}

OTHER_CATEGORICAL_FEATURES = {
    "publisher_id_hash",
    "source_item_type",
    "browser_platform",
    "os_family",
}

NUMERICAL_FEATURES = {
    "empiric_calibrated_recs",
    "empiric_clicks",
    "user_target_recs",
    "user_clicks",
    "user_recs"
}

TARGET_TAXONOMY_FIELD = "target_item_taxonomy"
USER_RECS_FIELD = "user_recs"
USER_TARGET_RECS_FIELD = "user_target_recs"
USER_CLICKS_FIELD = "user_clicks"
LABEL = "is_click"

class FeaturesProcessor(object):
    def __init__(self):
        # map between field and category name to the index in the one hot encoding og this feature
        self.ohe_index_mapping: Dict[str, Dict[str, int]] = None
        self.page_view_standartizer = Standartizer


def create_features(data: pd.DataFrame) -> Tuple[Sequence[str], np.ndarray]:

    features_components = []
    features_headers = []
    for field, threshold in FEATURES_MIN_THRESHOLDS.items():
        mapped_items = filter_by_occurrence(data, field, threshold)
        new_field_name = f"{field}_fl"
        data[new_field_name] = mapped_items
        header, ohe = to_one_hot_encoding(mapped_items, feature_prefix=field)
        features_components.append(ohe)
        features_headers.append(header)

    for field in OTHER_CATEGORICAL_FEATURES:
        if field in FEATURES_MIN_THRESHOLDS:
            continue

        header, ohe = to_one_hot_encoding(data[field], feature_prefix=field)
        features_components.append(ohe)
        features_headers.append(header)

    # process taxonomy categorical features
    taxonomy_headers, taxonomy_features = create_taxonomy_fearture(data[TARGET_TAXONOMY_FIELD])
    features_components.append(taxonomy_features)
    features_headers.append(taxonomy_headers)

    # process numeric featured
    numeric_headers, numeric_features = process_numeric_features(data, NUMERICAL_FEATURES)
    features_components.append(numeric_features)
    features_headers.append(numeric_headers)

    # process time features
    time_headers, time_features = process_time_features(data, NUMERICAL_FEATURES)
    features_components.append(time_features)
    features_headers.append(time_headers)

    features = np.hstack(features_components)
    headers = list(chain(*features_headers))
    return headers, features


def create_taxonomy_fearture(taxonomy: Sequence[str]) -> Tuple[Sequence[str], np.ndarray]:
    categories_by_hierarchy = [[], [], []]
    for tax in taxonomy:
        tax_categories = tax.split("~")
        for i, category in enumerate(tax_categories):
            categories_by_hierarchy[i].append(category)

    ohe_sets = []
    ohe_headers = []
    for i in range(len(categories_by_hierarchy)):
        categories = categories_by_hierarchy[i]
        prefix = f"tax{i}"
        header, ohe = to_one_hot_encoding(categories, feature_prefix=prefix)
        ohe_sets.append(ohe)
        ohe_headers.append(header)

    tax_features = np.hstack(ohe_sets)
    headers = list(chain(*ohe_headers))
    return headers, tax_features


def process_numeric_features(data: pd.DataFrame, numeric_fields: Set[str]) -> Tuple[Sequence[str], np.ndarray]:
    users_clicks_ratio = data[USER_CLICKS_FIELD].values / data[USER_RECS_FIELD].values
    user_target_recs_ratio = data[USER_TARGET_RECS_FIELD].values / data[USER_RECS_FIELD].values
    features = [users_clicks_ratio, user_target_recs_ratio]
    headers = ["users_clicks_ratio", "user_target_recs_ratio"]
    for field in numeric_fields:
        features.append(data[field].values)
        headers.append(field)

    return headers, np.hstack(features)


def process_time_features(data: pd.DataFrame, time_fields: Set[str]) -> Tuple[Sequence[str], np.ndarray]:
    pass









from itertools import chain
from operator import itemgetter
from typing import Tuple, Sequence, Set, Any, Dict, List

import numpy as np
import pandas as pd
import sklearn

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

NUMERIC_FEATURES = [
    "empiric_calibrated_recs",
    "empiric_clicks",
    "user_target_recs",
    "user_clicks",
    "user_recs"
]

TARGET_TAXONOMY_FIELD = "target_item_taxonomy"
NO_TAX_CATEGORY_VALUE = "NO_CATEGORY"
USER_RECS_FIELD = "user_recs"
USER_TARGET_RECS_FIELD = "user_target_recs"
USER_CLICKS_FIELD = "user_clicks"
VIEW_TIME_FIELD = "page_view_start_time"
DAY_FIELD = "day_of_week"
HOUR_FIELD = "time_of_day"
GMT_FIELD = "gmt_offset"
LABEL = "is_click"

NUM_DAYS_IN_WEEK = 7
WEEKEND_BEGIN_DAY = 6
NUM_PARTS_OF_DAY = 6

#parts of the day
EARLY_MORNING = 0
EARLY_MORNING_RANGE = range(2,6)

MORNING = 1
MORNING_RANGE = range(6, 11)

NOON = 2
NOON_RANGE = range(11, 15)

AFTERNOON = 3
AFTERNOON_RANGE = range(15, 19)

EVENING = 4
EVENING_RANGE = range(19, 23)

NIGHT = 5


class FeaturesProcessor(object):
    def __init__(self):
        # map between field and category name to the index in the one hot encoding og this feature
        self.ohe_index_mapping: Dict[str, Dict[str, int]] = None
        self.page_view_min_time: int = None
        self.features_names: List[str] = None
        self.__taxonomy_max_depth: int = None

    def fit(self, data: pd.DataFrame) -> 'FeaturesProcessor':
        self.ohe_index_mapping = {}
        headers = []
        for field, threshold in FEATURES_MIN_THRESHOLDS:
            index_mapping, header = create_ohe_index_with_header(data[field], field, threshold)
            self.ohe_index_mapping[field] = index_mapping
            headers.append(header)

        for field in OTHER_CATEGORICAL_FEATURES:
            assert(field not in FEATURES_MIN_THRESHOLDS)    # just in case, to prevent duplicated huge one hot feature!
            index_mapping, header = create_ohe_index_with_header(data[field], field)
            self.ohe_index_mapping[field] = index_mapping
            headers.append(header)

        index_mappings_by_depth, headers_by_depth = create_taxonomy_index(data[TARGET_TAXONOMY_FIELD])
        self.__taxonomy_max_depth = len(index_mappings_by_depth)
        for i in range(len(index_mappings_by_depth)):
            mapping_name = f"tax{i}"
            tax_index_mapping = index_mappings_by_depth[i]
            self.ohe_index_mapping[mapping_name] = tax_index_mapping
            headers.append(list(chain(*headers_by_depth)))

        numeric_headers = get_numeric_headers()
        headers.append(numeric_headers)

        self.page_view_min_time = min(data[VIEW_TIME_FIELD])
        temporal_headers = get_temporal_headers()
        headers.append(temporal_headers)

        self.features_names = list(chain(*headers))
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        features = []
        categorical_features = list(map(itemgetter(0), FEATURES_MIN_THRESHOLDS)) + OTHER_CATEGORICAL_FEATURES
        for field in categorical_features:
            index_mapping = self.ohe_index_mapping[field]
            ohe = create_ohe_features(data[field], index_mapping)
            features.append(ohe)

        # taxonomy features
        index_mappings_by_depth = []
        for i in range(self.__taxonomy_max_depth):
            index_mapping = self.ohe_index_mapping[f"tax{i}"]
            index_mappings_by_depth.append(index_mapping)

        ohe = create_taxonomy_features(data[TARGET_TAXONOMY_FIELD], index_mappings_by_depth)
        features.append(ohe)

        numeric_features = process_numeric_features(data, NUMERIC_FEATURES)
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


def create_taxonomy_features(taxonomy_values: Sequence[str], index_mappings_by_depth: List[Dict[str, int]]) -> np.ndarray:
    categories_by_hierarchy = get_categories_by_depth(taxonomy_values)

    ohe_sets = []
    for i in range(len(categories_by_hierarchy)):
        categories = categories_by_hierarchy[i]
        index_mapping = index_mappings_by_depth[i]
        ohe = create_ohe_features(categories, index_mapping)
        ohe_sets.append(ohe)

    tax_features = np.hstack(ohe_sets)
    return tax_features


def get_numeric_headers() -> List[str]:
    numeric_headers = ["users_clicks_ratio", "user_target_recs_ratio"]
    numeric_headers.extend(NUMERIC_FEATURES)
    return numeric_headers


def process_numeric_features(data: pd.DataFrame, numeric_fields: List[str]) -> np.ndarray:
    user_recs = data[USER_RECS_FIELD].values + 1

    users_clicks_ratio = data[USER_CLICKS_FIELD].values / user_recs
    users_clicks_ratio[user_recs == 0] = 0

    user_target_recs_ratio = data[USER_TARGET_RECS_FIELD].values / user_recs
    user_target_recs_ratio[user_recs == 0] = 0

    features = [users_clicks_ratio.reshape(-1, 1), user_target_recs_ratio.reshape(-1, 1)]
    features.append(data[numeric_fields].values)
    return np.hstack(features)


def get_temporal_headers():
    return ["ad_view_time"]\
           + get_weekday_feature_header()\
           + get_hours_feature_header()\
           + ["gmt_offset"]


def process_temporal_features(data: pd.DataFrame, min_view_time: int) -> np.ndarray:
    view_time_feature = process_view_time_features(data[VIEW_TIME_FIELD], min_view_time)
    weekday_features = process_week_days_features(data[DAY_FIELD])
    hour_features = process_hour_features(data[HOUR_FIELD])
    gmt_feature = process_gmt_offset_features(data[GMT_FIELD])
    return np.hstack((view_time_feature, weekday_features, hour_features, gmt_feature))


def process_view_time_features(values: Sequence[int], min_value: int) -> np.ndarray:
    return (np.asarray(values) - min_value).reshape(-1, 1)


def get_weekday_feature_header() -> Sequence[str]:
    return [f"day{i}" for i in range(NUM_DAYS_IN_WEEK)] + ["is_weekend"]


def process_week_days_features(values: Sequence[str]) -> np.ndarray:
    num_features = NUM_DAYS_IN_WEEK + 1 # plus another feature for 'is_weekend'
    ohe = np.zeros((len(values), num_features), dtype=bool)
    for i, day in enumerate(values):
        ohe[i][day] = True
        if i >= WEEKEND_BEGIN_DAY:
            ohe[i][-1] = True

    return ohe


def get_part_of_day_index(hour: int) -> int:
    if hour in EARLY_MORNING_RANGE:
        return EARLY_MORNING
    if hour in MORNING_RANGE:
        return MORNING
    if hour in NOON_RANGE:
        return NOON
    if hour in AFTERNOON_RANGE:
        return AFTERNOON
    if hour in EVENING_RANGE:
        return EVENING

    return NIGHT


def get_hours_feature_header() -> Sequence[str]:
    return [f"hour{i}" for i in range(NUM_DAYS_IN_WEEK)]\
           + ["is_early_morning", "is_morning", "is_noon", "is_afternoon", "is_evening", "is_night"]


def process_hour_features(values: Sequence[int]) -> np.ndarray:
    num_features = NUM_PARTS_OF_DAY + 1 # another feature contains the original jour values (as an ordinal variable).
    features = np.zeros((len(values), num_features), dtype=np.uint8)
    for i, hour in enumerate(values):
        day_part = get_part_of_day_index(hour)
        features[i][day_part] = True
        features[i][-1] = hour

    return features


def process_gmt_offset_features(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values).reshape(-1, 1)

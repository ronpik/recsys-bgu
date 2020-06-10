from typing import Sequence

import pandas as pd
import numpy as np

VIEW_TIME_FIELD = "page_view_start_time"
DAY_FIELD = "day_of_week"
HOUR_FIELD = "time_of_day"
GMT_FIELD = "gmt_offset"

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
        ohe[i][int(day)] = True
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
    return ["is_early_morning", "is_morning", "is_noon", "is_afternoon", "is_evening", "is_night", "hour"]


def process_hour_features(values: Sequence[int]) -> np.ndarray:
    num_features = NUM_PARTS_OF_DAY + 1 # another feature contains the original jour values (as an ordinal variable).
    features = np.zeros((len(values), num_features), dtype=np.uint8)
    for i, hour in enumerate(values):
        day_part = get_part_of_day_index(hour)
        features[i][day_part] = True
        features[i][-1] = int(hour)

    return features


def process_gmt_offset_features(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values).reshape(-1, 1)
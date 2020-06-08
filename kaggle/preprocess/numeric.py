from typing import List

import pandas as pd
import numpy as np

USER_RECS_FIELD = "user_recs"
USER_TARGET_RECS_FIELD = "user_target_recs"
USER_CLICKS_FIELD = "user_clicks"

NUMERIC_FEATURES = [
    "empiric_calibrated_recs",
    "empiric_clicks",
    "user_target_recs",
    "user_clicks",
    "user_recs"
]


def get_numeric_headers() -> List[str]:
    numeric_headers = ["users_clicks_ratio", "user_target_recs_ratio"]
    numeric_headers.extend(NUMERIC_FEATURES)
    return numeric_headers


def process_numeric_features(data: pd.DataFrame) -> np.ndarray:
    user_recs = data[USER_RECS_FIELD].values + 1

    users_clicks_ratio = data[USER_CLICKS_FIELD].values / user_recs
    users_clicks_ratio[user_recs == 0] = 0

    user_target_recs_ratio = data[USER_TARGET_RECS_FIELD].values / user_recs
    user_target_recs_ratio[user_recs == 0] = 0

    features = [users_clicks_ratio.reshape(-1, 1),
                user_target_recs_ratio.reshape(-1, 1),
                data[NUMERIC_FEATURES].values
                ]
    return np.hstack(features)
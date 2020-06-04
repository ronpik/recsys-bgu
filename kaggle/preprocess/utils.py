from collections import Counter
from itertools import takewhile
from operator import itemgetter
from typing import Any, Sequence, Tuple

import pandas as pd
import numpy as np

NON_FREQ_NAME = "unk"


def filter_by_occurrence(data: pd.DataFrame, field: str, min_count: int) -> pd.Series:
    counts = Counter(data[field])
    frequent_items = set(map(itemgetter(0), takewhile(lambda c: c[1] >= min_count, counts)))
    mapped_items = pd.Series(item if item in frequent_items else NON_FREQ_NAME for item in data[field])
    return mapped_items


def to_one_hot_encoding(feature_items: Sequence[Any], feature_prefix: str = "") -> Tuple[Sequence[str], np.ndarray]:
    """
    take a categorical feature (of Any type) and returns it perform one hot encoding
    :param feature_items:
    :return: a tuple of the names of the categories encoded feature followed by the encoded feature as one hot.
    """
    counts = Counter(feature_items)  # used to sort indices by frequency, not really necessary
    index_mapping = {item: i for i, item in enumerate(counts)}
    header = [f"{feature_prefix}:{name}" for name in counts]
    num_unique = len(index_mapping)
    ohe = np.zeros((len(feature_items), num_unique), dtype=bool)
    for i, item in enumerate(feature_items):
        item_index = index_mapping[item]
        ohe[i][item_index] = 1

    return header, ohe

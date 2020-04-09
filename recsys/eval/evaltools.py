from itertools import groupby
from operator import itemgetter
from typing import Sequence, NamedTuple, Any

import numpy as np


def rmse(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.array(x)
    y = np.array(y)
    score = np.sqrt(np.mean(np.square(x - y)))
    return score


class ConfusionMatrix(NamedTuple):
    tp: int
    fp: int
    fn: int
    tn: int


def get_confusion_matrix(y_true: Sequence[float], y_pred: Sequence[float]) -> ConfusionMatrix:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.logical_and(y_true, y_pred)
    fp = np.logical_and(~y_true, y_pred)
    fn = np.logical_and(y_true, ~y_pred)
    tn = np.logical_and(~y_true, ~y_pred)
    return ConfusionMatrix(tp, fp, fn, tn)


def average_ndpm(y_true: Sequence[float], y_pred: Sequence[float], users: Sequence[Any]) -> float:
    predictions = zip(users, y_true, y_pred)
    sorted_predictions = sorted(predictions, key=itemgetter(0))
    num_usres = 0
    sum_scores = 0
    for _, user_predictions in groupby(sorted_predictions, key=itemgetter(0)):
        _, user_true, user_pred = zip(*user_predictions)
        user_score = ndpm(user_true, user_pred)
        sum_scores += user_score
        num_usres += 1

    return sum_scores / num_usres


def ndpm(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    num_items = len(y_true)
    sum_distances = 0
    for i in range(num_items - 1):
        true_rank1 = y_true[i]
        pred_rank1 = y_pred[i]
        for j in range(i + 1, num_items):
            true_rank2 = y_true[j]
            pred_rank2 = y_pred[j]

            pair_distance = get_pair_order_distance(true_rank1, true_rank2, pred_rank1, pred_rank2)
            sum_distances += pair_distance

    num_pairs = 0.5 * num_items * (num_items + 1)
    return sum_distances / num_pairs


def get_pair_order_value(rank1, rank2) -> int:
    if rank1 == rank2:
        return 0
    if rank1 < rank2:
        return -1

    return 1


def get_pair_order_distance(true_rank1, true_rank2, pred_rank1, pred_rank2):
    true_orde_value = get_pair_order_value(true_rank1, true_rank2)
    if true_orde_value == 0:
        return 0

    pred_order_value = get_pair_order_value(pred_rank1, pred_rank2)
    if pred_order_value == 0:
        return 0.5

    if true_orde_value == pred_order_value:
        return 0

    return 1




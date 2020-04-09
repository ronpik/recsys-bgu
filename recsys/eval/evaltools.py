from typing import Sequence, NamedTuple

import numpy as np

def rmse(x: Sequence[float], y: Sequence[float]) -> float:
    sum_squares = np.sum(np.power(np.asarray(x) - np.asarray(y), 2))
    return np.sqrt(float(sum_squares) / len(x))


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



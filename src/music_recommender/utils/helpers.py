from collections import defaultdict

import numpy as np
from sklearn.feature_selection import (
    mutual_info_regression,
)


def tree():
    """Create nested defaultdict"""
    return defaultdict(tree)


def multi_output_mutual_info(X, y):
    if len(y.shape) == 1:
        return mutual_info_regression(X, y)

    mi_scores = np.zeros(X.shape[1])
    for i in range(y.shape[1]):
        mi_scores += mutual_info_regression(X, y[:, i])

    return mi_scores / y.shape[1]

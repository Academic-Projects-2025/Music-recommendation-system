import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_regression,
)

from src.music_recommender.config import Config

cfg = Config()


def multi_output_mutual_info(X, y):
    if len(y.shape) == 1:
        return mutual_info_regression(X, y)

    mi_scores = np.zeros(X.shape[1])
    for i in range(y.shape[1]):
        mi_scores += mutual_info_regression(X, y[:, i])

    return mi_scores / y.shape[1]



class ReduceNumFeature(BaseEstimator, TransformerMixin):
    def __init__(self, k=150, variance_thershold=0.95) -> None:
        super().__init__()
        self.k = k
        self.variance_filter = None
        self.correlation_columns_to_drop = None
        self.selector = None
        self.variance_thershold = variance_thershold

    def fit(self, X, y=None):
        self.variance_filter = VarianceThreshold(threshold=self.variance_thershold)
        X_filt_var = self.variance_filter.fit_transform(X)
        print(f"After variance filter: {X_filt_var.shape[1]} features")

        X_filt_var_df = pd.DataFrame(X_filt_var)
        correlation_matrix = X_filt_var_df.corr().abs()

        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        self.correlation_columns_to_drop = [
            column for column in upper.columns if any(upper[column] > 0.95)
        ]

        X_filt_uncorr = X_filt_var_df.drop(
            columns=self.correlation_columns_to_drop
        ).values
        print(f"After correlation filter: {X_filt_uncorr.shape[1]} features")

        if y is not None:
            k_actual = min(self.k, X_filt_uncorr.shape[1])
            self.selector = SelectKBest(multi_output_mutual_info, k=k_actual)
            self.selector.fit(X_filt_uncorr, y)
            print(
                f"After SelectKBest: {self.selector.transform(X_filt_uncorr).shape[1]} features"
            )

        return self

    def transform(self, X):
        X_filt_var = self.variance_filter.transform(X)

        X_filt_var_df = pd.DataFrame(X_filt_var)
        X_filt_uncorr = X_filt_var_df.drop(
            columns=self.correlation_columns_to_drop
        ).values

        if self.selector is not None:
            X_filt_selected = self.selector.transform(X_filt_uncorr)
        else:
            X_filt_selected = X_filt_uncorr

        return X_filt_selected


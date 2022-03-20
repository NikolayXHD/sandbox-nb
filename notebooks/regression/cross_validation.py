from __future__ import annotations

import typing

from joblib.memory import Memory
import numpy as np
from sklearn import base

# noinspection PyProtectedMember
from sklearn.model_selection._split import check_cv

# noinspection PyProtectedMember
from sklearn.utils.metaestimators import _safe_split

from . import histogram
from .memory import control_output


# noinspection PyPep8Naming
class CrossValidation:
    def __init__(
        self,
        memory_: Memory,
        test_histogram_bins: typing.Tuple[int, int] | None = None,
        verbose: bool = False,
        cache: bool = False,
    ):
        self.cache = cache
        if cache:
            self._splits_classifier_cache = control_output(
                memory_.cache(_splits_classifier, ignore=['estimator']),
                verbose=verbose,
            )
            self._create_histogram_cache = control_output(
                memory_.cache(histogram.create_histogram),
                verbose=verbose,
            )
        else:
            self._splits_classifier_cache = _splits_classifier
            self._create_histogram_cache = histogram.create_histogram

        self._test_histogram_bins = test_histogram_bins
        self._verbose = verbose

    def cross_validate(self, estimator, X, y, *, cv=None) -> np.ndarray:
        cv = check_cv(cv, y, classifier=False)
        test_scores = [
            self._fit_and_score_r2oos(
                base.clone(estimator),
                X_train,
                y_train,
                X_test,
                y_test,
            )
            for X_train, y_train, X_test, y_test in self._get_splits(
                estimator, X, y, cv
            )
        ]
        return np.array(test_scores)

    def _get_splits(
        self, estimator, X, y, cv
    ) -> typing.Iterable[
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        return self._splits_classifier_cache(
            estimator, base.is_classifier(estimator), X, y, cv
        )

    def _fit_and_score_r2oos(
        self, estimator, X_train, y_train, X_test, y_test
    ) -> float:
        if self._test_histogram_bins is None:
            return self._fit_and_score(
                X_test, X_train, estimator, y_test, y_train
            )
        else:
            return self._fit_and_score_hist(
                X_test, X_train, estimator, y_test, y_train
            )

    def _fit_and_score(self, X_test, X_train, estimator, y_test, y_train):
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        assert y_test.shape[0] == y_pred.shape[0]
        n_samples = X_test.shape[0]
        residual_sum_squares = ((y_test - y_pred) ** 2).sum(
            axis=0, dtype=np.float64
        ) / n_samples
        y_train_mean = np.average(y_train, axis=0)
        total_sum_squares = ((y_test - y_train_mean) ** 2).sum(
            axis=0, dtype=np.float64
        ) / n_samples
        return self._score(residual_sum_squares, total_sum_squares)

    def _fit_and_score_hist(self, X_test, X_train, estimator, y_test, y_train):
        estimator.fit(X_train, y_train)
        X_test_h, y_test_h, w_test_h = self._create_histogram_cache(
            X_test, y_test, None, self._test_histogram_bins, same_x=False
        )
        y_pred_h = estimator.predict(X_test_h)
        w_total = w_test_h.sum()
        residual_sum_squares = (((y_test_h - y_pred_h) ** 2) * w_test_h).sum(
            axis=0, dtype=np.float64
        ) / w_total
        y_train_mean = np.average(y_train, axis=0)
        total_sum_squares = (((y_test_h - y_train_mean) ** 2) * w_test_h).sum(
            axis=0, dtype=np.float64
        ) / w_total
        return self._score(residual_sum_squares, total_sum_squares)

    def _score(self, residual_sum_squares, total_sum_squares):
        return 1 - residual_sum_squares / total_sum_squares


# noinspection PyPep8Naming
def _splits_classifier(
    estimator, is_classifier, X, y, cv
) -> typing.Iterable[
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    splits = []
    for test, train in cv.split(X, y, None):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        splits.append((X_train, y_train, X_test, y_test))
    return splits


__all__ = ['CrossValidation']

from __future__ import annotations

import typing

import cupy
from joblib.memory import Memory
import numpy as np
from sklearn import base, model_selection

from . import histogram
from .memory import control_output


# noinspection PyPep8Naming
class CrossValidation:
    def __init__(
        self,
        memory_: Memory | None,
        test_histogram_bins: typing.Tuple[int, int] | None = None,
        regressor_supports_cupy: bool = True,
        verbose: bool = False,
        cache: bool = False,
    ):
        self._cache = cache
        if cache:
            assert memory_ is not None
            self._splits_cache = control_output(
                memory_.cache(_splits), verbose=verbose
            )
            self._create_histogram_cache = control_output(
                memory_.cache(histogram.create_histogram), verbose=verbose
            )
        else:
            assert memory_ is None
            self._splits_cache = _splits
            self._create_histogram_cache = histogram.create_histogram

        self._test_histogram_bins = test_histogram_bins
        self._regressor_supports_cupy = regressor_supports_cupy
        self._verbose = verbose

    def cross_validate(self, estimator, X, y, w, *, cv=None) -> np.ndarray:
        test_scores = [
            self._fit_and_score_r2oos(
                base.clone(estimator),
                X_train,
                y_train,
                w_train,
                X_test,
                y_test,
                w_test,
            )
            for (
                X_train,
                y_train,
                w_train,
                X_test,
                y_test,
                w_test,
            ) in self._get_splits(X, y, w, cv)
        ]
        return np.array(test_scores)

    def _get_splits(
        self, X, y, w, cv
    ) -> typing.Iterable[
        typing.Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        return self._splits_cache(X, y, w, cv)

    def _fit_and_score_r2oos(
        self, estimator, X_train, y_train, w_train, X_test, y_test, w_test
    ) -> float:
        if self._test_histogram_bins is None:
            return self._fit_and_score(
                X_test, X_train, estimator, y_test, y_train, w_test, w_train
            )
        else:
            return self._fit_and_score_hist(
                X_test, X_train, estimator, y_test, y_train, w_test, w_train
            )

    def _fit_and_score(
        self, X_test, X_train, estimator, y_test, y_train, w_test, w_train
    ):
        if not self._regressor_supports_cupy:
            X_train = X_train.get()
            y_train = y_train.get()
            if w_train is not None:
                w_train = w_train.get()
            X_test = X_test.get()
            y_test = y_test.get()
            if w_test is not None:
                w_test = w_test.get()

        if w_train is None:
            estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train, sample_weight=w_train)
        y_pred = estimator.predict(X_test)
        assert y_test.shape[0] == y_pred.shape[0]
        residual_squares = (y_test - y_pred) ** 2
        if w_test is not None:
            residual_squares *= w_test
        residual_sum_squares = residual_squares.sum(axis=0, dtype=np.float64)
        y_train_mean = np.average(y_train, weights=w_train, axis=0)
        total_squares = (y_test - y_train_mean) ** 2
        if w_test is not None:
            total_squares *= w_test
        total_sum_squares = total_squares.sum(axis=0, dtype=np.float64)
        return 1 - residual_sum_squares / total_sum_squares

    def _fit_and_score_hist(
        self, X_test, X_train, estimator, y_test, y_train, w_test, w_train
    ):
        if not self._regressor_supports_cupy:
            X_train = X_train.get()
            y_train = y_train.get()
            if w_train is not None:
                w_train = w_train.get()

        estimator.fit(X_train, y_train, sample_weight=w_train)
        X_test_h, y_test_h, w_test_h = self._create_histogram_cache(
            X_test,
            y_test,
            w_test,
            self._test_histogram_bins,
            same_x=False,
        )
        if not self._regressor_supports_cupy:
            X_test_h = X_test_h.get()
            y_test_h = y_test_h.get()
            w_test_h = w_test_h.get()
        y_pred_h = estimator.predict(X_test_h)
        residual_sum_squares = (((y_test_h - y_pred_h) ** 2) * w_test_h).sum(
            axis=0, dtype=np.float64
        )
        y_train_mean = np.average(y_train, weights=w_train, axis=0)
        total_sum_squares = (((y_test_h - y_train_mean) ** 2) * w_test_h).sum(
            axis=0, dtype=np.float64
        )
        return 1 - residual_sum_squares / total_sum_squares


# noinspection PyPep8Naming
def _splits(
    X: cupy.ndarray,
    y: cupy.ndarray,
    w: cupy.ndarray,
    cv: model_selection.BaseCrossValidator,
) -> typing.Iterable[
    typing.Tuple[
        cupy.ndarray,
        cupy.ndarray,
        cupy.ndarray,
        cupy.ndarray,
        cupy.ndarray,
        cupy.ndarray,
    ]
]:
    splits = []
    for test, train in cv.split(X, y):
        test_cu = cupy.array(test)
        train_cu = cupy.array(train)
        X_train = X[train_cu, ...]
        X_test = X[test_cu, ...]

        y_train = y[train_cu]
        y_test = y[test_cu]

        if w is None:
            w_train = None
            w_test = None
        else:
            w_train = w[train]
            w_test = w[test]

        splits.append((X_train, y_train, w_train, X_test, y_test, w_test))
    return splits


__all__ = ['CrossValidation']

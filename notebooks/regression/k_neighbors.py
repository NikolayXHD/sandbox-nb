from __future__ import annotations

import warnings

import numpy as np

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

# noinspection PyProtectedMember
from sklearn.neighbors._base import _get_weights


# noinspection PyPep8Naming
# noinspection PyMethodOverriding
class KNeighborsWeightedRegressor(KNeighborsRegressor):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self._sample_weight: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            assert sample_weight.shape == (X.shape[0],)
            assert sample_weight.min() >= 0
        self._sample_weight = sample_weight
        super().fit(X, y)
        return self

    def predict(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        sample_weight = self._sample_weight
        if weights is None and sample_weight is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            if weights is None:
                weights = sample_weight[neigh_ind]
            elif sample_weight is not None:
                weights *= sample_weight[neigh_ind]

            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


# noinspection PyPep8Naming
# noinspection PyMethodOverriding
class RadiusNeighborsWeightedRegressor(RadiusNeighborsRegressor):
    def fit(self, X, y, sample_weight=None):
        # noinspection PyAttributeOutsideInit
        self._sample_weight = sample_weight
        super().fit(X, y)
        return self

    def predict(self, X):
        neigh_dist, neigh_ind = self.radius_neighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        empty_obs = np.full_like(_y[0], np.nan)

        sample_weight = self._sample_weight
        if weights is None and sample_weight is None:
            y_pred = np.array(
                [
                    np.mean(_y[ind, :], axis=0) if len(ind) else empty_obs
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )
        elif weights is None:
            y_pred = np.array(
                [
                    np.average(
                        _y[ind, :],
                        axis=0,
                        weights=sample_weight[neigh_ind[i]],
                    )
                    if len(ind)
                    else empty_obs
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )
        elif sample_weight is None:
            y_pred = np.array(
                [
                    np.average(_y[ind, :], axis=0, weights=weights[i])
                    if len(ind)
                    else empty_obs
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )
        else:
            y_pred = np.array(
                [
                    np.average(
                        _y[ind, :],
                        axis=0,
                        weights=weights[i] * sample_weight[neigh_ind[i]],
                    )
                    if len(ind)
                    else empty_obs
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )

        if np.any(np.isnan(y_pred)):
            empty_warning_msg = (
                'One or more samples have no neighbors '
                'within specified radius; predicting NaN.'
            )
            warnings.warn(empty_warning_msg)

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


__all__ = ['KNeighborsWeightedRegressor', 'RadiusNeighborsWeightedRegressor']

from __future__ import annotations

import typing

from joblib.memory import Memory
import cupy
import cudf
from sklearn.preprocessing import FunctionTransformer

from .memory import control_output


class Histogram2dRegressionWrapper:
    def __init__(
        self,
        regressor,
        bins: typing.Tuple[int, int],
        memory_: Memory,
        *,
        transformer: FunctionTransformer | None = None,
        same_x: bool = True,
        regressor_supports_cupy: bool = True,
        verbose: bool = False,
        cache: bool = False,
    ):
        self.regressor = regressor
        self.bins = bins
        self.same_x = same_x
        self.regressor_supports_cupy = regressor_supports_cupy
        self.memory_ = memory_
        self.transformer = transformer
        self.verbose = verbose
        self.cache = cache
        if cache:
            self._create_histogram_cache = control_output(
                memory_.cache(create_histogram), verbose=verbose
            )
        else:
            self._create_histogram_cache = create_histogram

    # noinspection PyUnusedLocal
    def get_params(self, deep):
        return {
            'regressor': self.regressor,
            'bins': self.bins,
            'same_x': self.same_x,
            'regressor_supports_cupy': self.regressor_supports_cupy,
            'memory_': self.memory_,
            'transformer': self.transformer,
            'verbose': self.verbose,
            'cache': self.cache,
        }

    # noinspection PyPep8Naming
    def fit(self, X, y, sample_weight=None):
        assert len(self.bins) == X.shape[1] + 1
        assert len(y.shape) == 1

        if sample_weight is not None:
            assert sample_weight.shape == (
                X.shape[0],
            ), f'{sample_weight.shape=}, {X.shape=}'

        if self.transformer is not None:
            X = self.transformer.transform(X)

        X_hist, y_hist, w_hist = self._get_histogram(
            X, y, sample_weight, self.bins, same_x=self.same_x
        )
        if not self.regressor_supports_cupy:
            X_hist = X_hist.get()
            y_hist = y_hist.get()
            w_hist = w_hist.get()
        if self.transformer is not None:
            X_hist = self.transformer.inverse_transform(X_hist)

        self.regressor = self.regressor.fit(
            X_hist, y_hist, sample_weight=w_hist
        )
        return self

    # noinspection PyPep8Naming
    def predict(self, X):
        return self.regressor.predict(X)

    # noinspection PyPep8Naming
    def _get_histogram(
        self,
        X,
        y,
        sample_weight,
        bins: typing.Tuple[int, int],
        same_x: bool,
    ):
        return self._create_histogram_cache(
            X, y, sample_weight, bins, same_x=same_x
        )


# noinspection PyPep8Naming
def create_histogram(
    X: cupy.ndarray,
    y: cupy.ndarray,
    sample_weight: cupy.ndarray | None,
    bins: typing.Tuple[int, ...],
    *,
    same_x: bool,
) -> typing.Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
    num_samples, num_features = X.shape
    assert num_samples > 0
    assert num_features > 0

    assert y.shape == (num_samples,)

    assert isinstance(bins, tuple)
    assert len(bins) == num_features + 1

    if sample_weight is not None:
        assert sample_weight.shape == (num_samples,)

    (*nums_x, num_y) = bins

    # noinspection PyArgumentList
    X_min = X.min(axis=0)
    # noinspection PyArgumentList
    X_max = X.max(axis=0)
    # noinspection PyArgumentList
    y_min = y.min()
    # noinspection PyArgumentList
    y_max = y.max()

    df = cudf.DataFrame(X)
    df.rename(
        {feat_i: f'X_{feat_i}' for feat_i in range(num_features)},
        axis=1,
        inplace=True,
    )
    for feat_i, num_x in enumerate(nums_x):
        if num_x > 1:
            X_delta = (X_max[feat_i] - X_min[feat_i]) / (num_x - 1)
            i_z_float = (X[:, feat_i] - X_min[feat_i]) / X_delta + 0.5
            i_z = i_z_float.astype('int')
        else:
            assert num_x == 1
            i_z = cupy.zeros((num_samples,), dtype='int')
        df[f'i_{feat_i}'] = i_z

    df['y'] = y
    if num_y > 1:
        y_delta = (y_max - y_min) / (num_y - 1)
        j_z = ((y - y_min) / y_delta + 0.5).astype('int')
    else:
        assert num_y == 1
        j_z = cupy.zeros(y.shape, dtype='int')
    df['j_z'] = j_z

    if sample_weight is None:
        w = cudf.Series(cupy.ones((num_samples,)), dtype='float')
    else:
        w = cudf.Series(sample_weight, dtype='float')
    df['w'] = w

    for feat_i, num_x in enumerate(nums_x):
        df[f'X_w_{feat_i}'] = df[f'X_{feat_i}'] * df['w']

    df['y_w'] = df['y'] * df['w']

    if same_x:
        grp_x = df.groupby([f'i_{feat_i}' for feat_i in range(num_features)])
        df_pivot_x_right = grp_x.agg(
            {f'i_{feat_i}': 'first' for feat_i in range(num_features)}
        )
        df_pivot_x_left = cudf.DataFrame()
        for feat_i in range(num_features):
            df_pivot_x_left[f'X_{feat_i}'] = (
                grp_x[f'X_w_{feat_i}'].sum() / grp_x['w'].sum()
            )
        df_pivot_x = cudf.concat([df_pivot_x_left, df_pivot_x_right], axis=1)

        # lets build i_z_to_X so that
        # i_z_to_X[i_0, i_1, ...] == [X_0, X_1, ...]

        #                                       i_0 .. i_n
        #                                                        0..a -> a+1 v
        dimension_to_num_i = df_pivot_x.iloc[:, num_features:].max(axis=0) + 1
        i_z_to_X = cupy.zeros((*dimension_to_num_i.values_host, num_features))

        # arr[(
        #   [1, 3, 5],
        #   [2, 4, 6]
        # )] = array([
        #   [10, 11],
        #   [12, 13],
        #   [14, 15],
        # ])
        #
        # is equivalent to
        #
        # arr[1, 2] = [10, 11]
        # arr[3, 4] = [12, 13]
        # arr[5, 6] = [13, 14]
        i_z_to_X[
            # == (i_0_values, i_1_values, ...)
            #                        i_0 .. i_n
            tuple(df_pivot_x.iloc[:, num_features:].T.values)
            # == (X_0_values, X_1_values, ...).T
            #                  X_0 .. X_n
        ] = df_pivot_x.iloc[:, :num_features].values

        grp = df.groupby(
            [*(f'i_{feat_i}' for feat_i in range(num_features)), 'j_z'],
            # #slow_sort_true
            # https://docs.rapids.ai/api/cudf/nightly/basics/groupby.html
            # sort=True is slow according to documentation, however while
            # testing performance slowdown was insignificant:
            # numpy + pandas: 30 s
            # cuda, sort=True: 7.5 s
            # cuda, sort=False: 6.5 s
            #
            # for reference, when histogram is loaded from cache
            # numpy + pandas: 2.5 s
            # cuda: 3 s
            sort=True,
        )
        df_pivot = grp.agg(
            {f'i_{feat_i}': 'first' for feat_i in range(num_features)},
        )

        #                                     i_0 ... i_n
        X_d = i_z_to_X[tuple(df_pivot.iloc[:, :num_features].T.values)]
        w_sum = grp['w'].sum()
        y_d = (grp['y_w'].sum() / w_sum).values
        w = w_sum.values
    else:
        grp = df.groupby(
            [*(f'i_{feat_i}' for feat_i in range(num_features)), 'j_z'],
            # #slow_sort_true
            sort=True,
        )
        df_pivot = cudf.DataFrame()
        w_sum = grp['w'].sum()
        for feat_i in range(num_features):
            df_pivot[f'X_{feat_i}'] = grp[f'X_w_{feat_i}'].sum() / w_sum
        #                      X_0 ... X_n
        X_d = df_pivot.iloc[:, :num_features].values
        y_d = (grp['y_w'].sum() / w_sum).values
        w = w_sum.values

    assert X_d.shape[0] == y_d.shape[0] == w.shape[0]
    return X_d, y_d, w


__all__ = ['Histogram2dRegressionWrapper', 'create_histogram']

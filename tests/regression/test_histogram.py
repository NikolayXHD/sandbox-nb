from __future__ import annotations

import itertools
import typing

import pytest
import numpy as np
from numpy import testing

from notebooks.regression import histogram


# noinspection PyPep8Naming
@pytest.mark.parametrize(
    ('X_1d', 'y', 'sample_weight', 'bins', 'same_x', 'result'),
    [
        *(
            (
                [10, 11, 12, 13],  # X
                [20, 21, 22, 23],  # y
                sample_weight,
                (2, 1),
                same_x,
                (
                    [10.5, 12.5],
                    [20.5, 22.5],
                    [2.0, 2.0],
                ),
            )
            # since bins.y == 1 same_x does not affect result
            for same_x in (False, True)
            for sample_weight in (None, [1, 1, 1, 1])
        ),
        *(
            (
                [10, 11, 12, 13],  # X
                [20, 21, 22, 23],  # y
                [30, 10, 30, 10],  # sample_weight
                (2, 1),
                same_x,
                (
                    [10.25, 12.25],
                    [20.25, 22.25],
                    [40.00, 40.00],
                ),
            )
            # since bins.y == 1 same_x does not affect result
            for same_x in (False, True)
        ),
        *itertools.chain(
            *(
                (
                    (
                        [10, 11, 12, 13],  # X
                        [20, 21, 22, 23],  # y
                        sample_weight,
                        (1, 2),
                        # same_x
                        False,
                        (
                            [10.5, 12.5],
                            [20.5, 22.5],
                            [2.0, 2.0],
                        ),
                    ),
                    (
                        [10, 11, 12, 13],  # X
                        [20, 21, 22, 23],  # y
                        sample_weight,
                        (1, 2),
                        # same_x
                        True,
                        (
                            [11.5, 11.5],
                            [20.5, 22.5],
                            [2.0, 2.0],
                        ),
                    ),
                )
                for sample_weight in (None, [1, 1, 1, 1])
            )
        ),
        (
            [10, 11, 12, 13],  # X
            [20, 21, 22, 23],  # y
            [30, 10, 30, 10],  # sample_weight
            (1, 2),
            # same_x
            False,
            (
                [10.25, 12.25],
                [20.25, 22.25],
                [40.00, 40.00],
            ),
        ),
        (
            [10, 11, 12, 13],  # X
            [20, 21, 22, 23],  # y
            [30, 10, 30, 10],  # sample_weight
            (1, 2),
            # same_x
            True,
            (
                [11.25, 11.25],
                [20.25, 22.25],
                [40.00, 40.00],
            ),
        ),
    ],
)
def test_result(
    X_1d: list[float],
    y: list[float],
    sample_weight: list[float] | None,
    bins: typing.Tuple[int, int],
    same_x: bool,
    result: typing.Tuple[list[float], list[float], list[float]],
):
    X_h_1d, y_h, w_h = result
    X_h = np.array(X_h_1d).reshape(-1, 1)

    X = np.array(X_1d).reshape(-1, 1)
    actual_result = histogram.create_histogram(
        X,
        np.array(y),
        None if sample_weight is None else np.array(sample_weight),
        bins,
        same_x=same_x,
    )
    X_h_actual, y_h_actual, w_h_actual = actual_result
    testing.assert_allclose(X_h_actual, X_h)
    testing.assert_allclose(y_h_actual, y_h)
    testing.assert_allclose(w_h_actual, w_h)


@pytest.mark.parametrize(
    ('X_2d', 'y_func', 'bins', 'X_h', 'y_h', 'w_h'),
    [
        (
            # X represents all points of 2d coordinate grid
            # from 0 to 1.5 with step 0.5
            # [[0.0, 0.0],
            #  [0.0, 0.5],
            #  ...
            #  [1.5, 1.5]]
            # X_2d
            np.mgrid[0:2:0.5, 0:2:0.5].reshape(2, -1).T,  # type: ignore[misc]
            # y_func
            (lambda X: X[:, 0]),
            # bins
            (2, 2, 2),
            # X_h
            np.array([[0.25, 0.25], [0.25, 1.25], [1.25, 0.25], [1.25, 1.25]]),
            # y_h
            np.array([0.25, 0.25, 1.25, 1.25]),
            # w_h
            np.array([4.0, 4.0, 4.0, 4.0]),
        ),
        (
            # X_2d
            np.mgrid[0:2:0.5, 0:2:0.5].reshape(2, -1).T,  # type: ignore[misc]
            # y_func
            (lambda X: X[:, 1]),
            # bins
            (2, 2, 2),
            # X_h
            np.array([[0.25, 0.25], [0.25, 1.25], [1.25, 0.25], [1.25, 1.25]]),
            # y_h
            np.array([0.25, 1.25, 0.25, 1.25]),
            # w_h
            np.array([4.0, 4.0, 4.0, 4.0]),
        ),
    ],
)
def test_2d_result(
    X_2d: np.ndarray,
    y_func: typing.Callable[[np.ndarray], np.ndarray],
    bins: typing.Tuple[int, int, int],
    X_h: np.ndarray,
    y_h: np.ndarray,
    w_h: np.ndarray,
):
    y = y_func(X_2d)
    X_h_actual, y_h_actual, w_h_actual = histogram.create_histogram(
        X_2d, y, None, bins=bins, same_x=True
    )
    testing.assert_allclose(X_h_actual, X_h)
    testing.assert_allclose(y_h_actual, y_h)
    testing.assert_allclose(w_h_actual, w_h)

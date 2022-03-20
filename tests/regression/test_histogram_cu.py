from __future__ import annotations

import itertools
import typing

import pytest
import cupy
from cupy import testing

from notebooks.regression_cu import histogram


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
    X_h = cupy.array(X_h_1d).reshape(-1, 1)

    X = cupy.array(X_1d).reshape(-1, 1)
    actual_result = histogram.create_histogram(
        X,
        cupy.array(y),
        None if sample_weight is None else cupy.array(sample_weight),
        bins,
        same_x=same_x,
    )
    X_h_actual, y_h_actual, w_h_actual = actual_result
    testing.assert_allclose(X_h_actual, X_h)
    testing.assert_allclose(y_h_actual, y_h)
    testing.assert_allclose(w_h_actual, w_h)

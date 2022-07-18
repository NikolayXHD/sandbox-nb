from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from regression.balance import get_w, get_weight_total_per_day

_1_DAY = 3600 * 24


@pytest.mark.parametrize(
    ('v_day', 'v_ticker', 'v_result'),
    [
        pytest.param(
            [1, 1, 2],  # day
            ['x', 'y', 'x'],  # ticker
            [0.5, 0.5, 1],  # expected result
            id='balances days to have equal weight'
        ),
        pytest.param(
            [1] * 99 + [1] * 9 + [2],  # day
            ['x'] * 99 + ['y'] * 9 + ['x'],  # ticker
            (
                # day 1 ticker x
                99 * [np.log1p(99) / 99 / (np.log1p(99) + np.log1p(9))]
                # day 1 ticker y
                + 9 * [np.log1p(9) / 9 / (np.log1p(99) + np.log1p(9))]
                # day 2 ticker x
                + [1]
            ),  # expected result
            id='rebalances tickers within day to have total log1p weight'
        )
    ]
)
def test_get_w(
    v_day: list[int], v_ticker: list[str], v_result: list[float]
):
    actual_result = get_w(
        t=pd.Series(_1_DAY * np.array(v_day), name='t'),
        ticker=pd.Series(v_ticker, name='ticker'),
    )
    assert_allclose(actual_result, np.array(v_result))


@pytest.mark.parametrize(
    ('v_day', 'v_w', 'v_result'),
    [
        (
            [1, 1, 1, 2, 2, 3, 4],
            [1, 1, 0, 0, 1, 3, 0],
            [2, 2, 2, 1, 1, 3, 0],
        ),
    ],
)
def test_get_weight_total_per_day(
    v_day: list[int], v_w: list[float], v_result: list[float]
):
    actual_result = get_weight_total_per_day(
        t=pd.Series(_1_DAY * np.array(v_day), name='t'),
        w=pd.Series(v_w, name='w'),
    )
    assert_allclose(actual_result, np.array(v_result))

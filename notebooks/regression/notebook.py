# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% cellId="r60bke4u6ugchb8rkiobw"
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import typing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from joblib.memory import Memory
from sklearn import model_selection

from regression import cross_validation
from regression import histogram
from regression.load import build_df, filter_df_by_dates

PWD = Path(os.path.dirname(os.path.realpath('__file__')))
CACHE_STORAGE_PATH = PWD.joinpath('..', '..', '.storage', 'cache')
OUTPUT_STORAGE_PATH = PWD.joinpath(
    '..', '..', '..', 'sandbox', '.storage', 'output'
)

durations = ('4h', '3d', '24d', '72d')
delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2022-07-01',
        '_'.join(durations),
        'market_False',
        'profit_currency_USD',
        'adv_True',
        'ad_True',
        'dlnv_True',
        'dln_True',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

delay_to_df_validate_raw = {
    delay: build_df(path, max_list_level=2)
    for delay, path in delay_to_dir.items()
}


# %%
def calculate_log_indicators(
    delay_to_df: dict[int, pd.DataFrame]
) -> dict[int, pd.DataFrame]:
    return {
        delay: append_log_indicators(df) for delay, df in delay_to_df.items()
    }


def append_log_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        **{
            f'{indicator}_log_{duration}': log_scale_value(
                df[f'{indicator}_{duration}'], scale
            )
            for duration in durations
            for indicator, scale in zip(
                ('adv', 'ad', 'dlnv', 'dln'),
                (3, 3, 1000, 1000),
            )
            if f'{indicator}_{duration}' in df
        }
    )
    i1 = df['dln_log_3d'] / 0.7
    i2 = df['dln_log_24d'] / 0.6
    hyperbolic_score = i2 ** 2 - i1 ** 2
    df = df.assign(**{'hyperbolic_score': hyperbolic_score})
    return df


def log_scale_value(values: np.ndarray, scale: float) -> np.ndarray:
    return np.sign(values) * np.log1p(scale * np.abs(values)) / np.log1p(scale)


def get_df(
    *,
    delay: int,
    date_from: datetime | None,
    date_to: datetime | None,
    use_validation_df: bool = False,
) -> pd.DataFrame:
    delay_to_df_map = (
        delay_to_df_validate if use_validation_df else delay_to_df
    )
    df = delay_to_df_map[delay]
    df = filter_df_by_dates(df, date_from, date_to)
    return df


def iterate_date_ranges(
    *,
    append_empty_range: bool = False,
    use_validation_df: bool = False,
) -> typing.Iterable[typing.Tuple[datetime | None, datetime | None]]:
    ranges = DATE_RANGES_VALIDATION if use_validation_df else DATE_RANGES
    for r in ranges:
        yield r
    if append_empty_range:
        yield (None, None)


def format_date(d: datetime | None) -> str:
    return str(d.date()) if d is not None else '***'


# Full dataset, including most recent year.
# Never use it to find regularities in data, rather use it to
# validate the findings
delay_to_df_validate = calculate_log_indicators(delay_to_df_validate_raw)

DATE_RANGES_VALIDATION = tuple(
    (datetime(y, 6, 1, 0, 0), datetime(y + 1, 6, 1, 0, 0))
    for y in range(2015, 2022)
)
DATE_RANGES = DATE_RANGES_VALIDATION[:-1]

delay_to_df = {
    delay: filter_df_by_dates(df, None, DATE_RANGES[-1][1])
    for delay, df in delay_to_df_validate.items()
}

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

# %%
delay_to_df[7]

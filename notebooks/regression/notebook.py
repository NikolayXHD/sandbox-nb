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

DATE_RANGES_VALIDATION = tuple(
    (datetime(y, 6, 1, 0, 0), datetime(y + 1, 6, 1, 0, 0))
    for y in range(2015, 2022)
)
DATE_RANGES = DATE_RANGES_VALIDATION[:-1]


delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2022-07-01',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

durations = ('3d', '24d', '72d')
# durations = ('4h', '3d', '24d', '72d')
indicators = ('dlnv', 'dln')
# indicators = ['dlnv', 'dln', 'adv', 'ad']

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)
memory_ = Memory(str(CACHE_STORAGE_PATH), mmap_mode='r', verbose=False)
delay_to_style = {7: ':', 30: '--', 180: '-'}


def update_delay_to_df():
    for delay, df in delay_to_df_validate.items():
        delay_to_df[delay] = filter_df_by_dates(df, None, DATE_RANGES[-1][1])


def get_df(
    *,
    delay: int,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
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
delay_to_df_validate = {
    delay: build_df(
        path,
        max_list_level=2,
        fields=[
            't',
            'ticker',
            'profit_in_currency',
            'w',
            *(
                f'{indicator}_{duration}'
                for indicator in indicators
                for duration in durations
            ),
        ],
    )
    for delay, path in delay_to_dir.items()
}

delay_to_df: dict[int, pd.DataFrame] = {}
update_delay_to_df()


# %%
def append_log_indicators(df: pd.DataFrame) -> None:
    for duration in durations:
        for indicator, scale in zip(
            ('adv', 'ad', 'dlnv', 'dln'), (3, 3, 1000, 1000)
        ):
            if f'{indicator}_{duration}' in df:
                df.loc[:, f'{indicator}_log_{duration}'] = log_scale_value(
                    df[f'{indicator}_{duration}'], scale
                )
                df.drop(f'{indicator}_{duration}', axis=1, inplace=True)


def log_scale_value(values: np.ndarray, scale: float) -> np.ndarray:
    return np.sign(values) * np.log1p(scale * np.abs(values)) / np.log1p(scale)


# Full dataset, including most recent year.
# Never use it to find regularities in data, rather use it to
# validate the findings
for df in delay_to_df_validate.values():
    append_log_indicators(df)
update_delay_to_df()


# %%
def append_score_dln_3d_24d(df: pd.DataFrame) -> None:
    i1 = df['dln_log_3d'] / 0.7
    i2 = df['dln_log_24d'] / 0.6
    hyperbolic_score = i2 ** 2 - i1 ** 2
    df.loc[:, 'score-dln-3d-24d'] = hyperbolic_score


for df in delay_to_df_validate.values():
    append_score_dln_3d_24d(df)
update_delay_to_df()


# %%
def append_score_dln_3d_72d(df: pd.DataFrame) -> None:
    i1 = df['dln_log_3d'] / 0.7
    i2 = df['dln_log_72d'] / 0.6
    hyperbolic_score = i2 ** 2 - i1 ** 2
    df.loc[:, 'score-dln-3d-72d'] = hyperbolic_score


for df in delay_to_df_validate.values():
    append_score_dln_3d_72d(df)
update_delay_to_df()

# %%
for delay, path in delay_to_dir.items():
    score_fields = ['dln-3d-dln-24d-0-exp-3d', 'dln-3d-dln-24d-0-exp-7d']
    df_scores = build_df(path, max_list_level=2, fields=['t', *score_fields])
    df = delay_to_df_validate[delay]
    for score_field in score_fields:
        df.loc[:, score_field] = df_scores.loc[:, score_field]

update_delay_to_df()

# %%
delay_to_df[7]

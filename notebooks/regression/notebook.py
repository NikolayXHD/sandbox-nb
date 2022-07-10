# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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

delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

durations = ('4h', '3d', '24d')

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
        'cmf_False',
        'ad_False',
        'dln_True',
        'dln_no_vol_True',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

# Full dataset, including most recent year.
# Never use it to find regularities in data, rather use it to
# validate the findings
delay_to_df_validate = {
    delay: build_df(path, max_list_level=2)
    for delay, path in delay_to_dir.items()
}

DATE_RANGES_VALIDATION = tuple(
    (datetime(y, 6, 1, 0, 0), datetime(y + 1, 6, 1, 0, 0))
    for y in range(2015, 2022)
)
DATE_RANGES = DATE_RANGES_VALIDATION[:-1]

delay_to_df = {
    delay: filter_df_by_dates(
        df, None, DATE_RANGES[-1][1]
    )
    for delay, df in delay_to_df_validate.items()
}

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)


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
    *, append_empty_range: bool = False, use_validation_df: bool = False,
) -> typing.Iterable[typing.Tuple[datetime | None, datetime | None]]:
    ranges = DATE_RANGES_VALIDATION if use_validation_df else DATE_RANGES
    for r in ranges:
        yield r
    if append_empty_range:
        yield (None, None)


def format_date(d: datetime | None) -> str:
    return str(d.date()) if d is not None else '***'


# %%
delay_to_df[180]

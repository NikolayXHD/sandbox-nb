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

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from joblib.memory import Memory
from sklearn import model_selection

from notebooks.regression import cross_validation
from notebooks.regression import histogram


def build_df(directory: Path):
    parts: list[pd.DataFrame] = []
    last_msg: str | None = None
    files = sorted(directory.glob('*.feather'))
    tickers = [f.stem for f in files]
    
    print(directory.resolve())
    for i, f in enumerate(files):
        if last_msg is not None:
            print('\r' + ' ' * len(last_msg) + '\r', end='')
        last_msg = f'{f.name} {i + 1} / {len(files)}'
        print(last_msg, end='')
        df_specific_ticker = pd.read_feather(f)
        df_specific_ticker['ticker'] = pd.Categorical(
            [f.stem] * len(df_specific_ticker), categories=tickers
        )
        parts.append(df_specific_ticker)
    print()
    df = pd.concat(parts)
    df.sort_values(by=['t'], inplace=True)
    return df


delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

PWD = Path(os.path.dirname(os.path.realpath('__file__')))
CACHE_STORAGE_PATH = PWD.joinpath('..', '..', '.storage', 'cache')
OUTPUT_STORAGE_PATH = PWD.joinpath(
    '..', '..', '..', 'sandbox', '.storage', 'output'
)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

durations = ('4h', '3d', '24d')

delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2021-07-01',
        '_'.join(durations),
        'market_False',
        'profit_currency_USD',
        'ad_True',
        'dln_True',
        'dln_no_vol_True',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

delay_to_df = {delay: build_df(path) for delay, path in delay_to_dir.items()}

# %%
min_v_01d = 0.25
min_v_72d = 0.05

month = 6

for delay in (7, 30, 180):
    print(delay)
    for year in (2015, 2016, 2017, 2018, 2019, 2020):
        df_all = delay_to_df[delay]
        print(f'    {year}.{month} -- {year + 1}.{month}')
        time_mask = df_all['t'].between(
            datetime(year, month, 1, 0, 0).timestamp(),
            datetime(year + 1, month, 1, 0, 0).timestamp() - 1,
        )
        df = df_all[time_mask]

        hot_mask = (df['indicator_4h'] < -min_v_01d) & (
            df['indicator_72d'] > min_v_72d
        )
        col_mask = (df['indicator_4h'] > +min_v_01d) & (
            df['indicator_72d'] > min_v_72d
        )

        print(
            f'    hot:  {df[hot_mask]["profit"].mean():.2f}, '
            f'freq: {hot_mask.sum() / len(df):.3f}'
        )
        print(
            f'    cold: {df[col_mask]["profit"].mean():.2f}, '
            f'freq: {col_mask.sum() / len(df):.3f}'
        )
        print(
            f'    h+c:  {df[col_mask|hot_mask]["profit"].mean():.2f}, '
            f'freq: {(col_mask.sum() + hot_mask.sum()) / len(df):.3f}'
        )
        print(f'    oth:  {df[~(col_mask|hot_mask)]["profit"].mean():.2f}')
        print()

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
    df = df.assign(**{'w': get_w(df)})
    return df


def get_w(df):
    """
    Formula for weight is explained in balancing.py notebook
    It satisfies 2 goals
    1. sum of weights per 1 day is always same (== 1)
    2. sum of weights of specific ticker for a given day is proportinal
       to logarightm of number of candles (n_samples). Without weight it
       would be proportional to number of candles (without logarithm)
    
    summary:
    1. all days have same weight, perfectly balanced
    2. tickers within day are less unbalanced
    
    Why not perfectly balance 2. as well?
    to reflect the fact it's still harder to buy / sell when
    number of candles is smaller
    """
    df_agg_source = df[['ticker', 't']]
    df_agg_source = df_agg_source.assign(
        **{'d': df_agg_source['t'] // (3600 * 24)}
    )
    
    df_agg_day_ticker = df_agg_source.groupby(
        ['d', 'ticker']
    ).agg(
        **{'n_td': ('t', 'count')}
    ).reset_index()

    df_agg_day_ticker = df_agg_day_ticker.assign(
        **{'ln_n_td': np.log(1 + df_agg_day_ticker['n_td'])}
    )
    
    df_agg_day = df_agg_day_ticker.groupby('d').agg(
        **{'sum_t_ln_n_td': ('ln_n_td', 'sum')}
    ).reset_index()
    
    df_agg_result = df_agg_source.merge(
        df_agg_day, on='d', copy=False
    ).merge(
        df_agg_day_ticker, on=('d', 'ticker'), copy=False
    )
    
    return (
        df_agg_result['ln_n_td']
        / df_agg_result['n_td']
        / df_agg_result['sum_t_ln_n_td'] 
    )


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

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

DATE_RANGES = (
    (datetime(2015, 6, 1, 0, 0), datetime(2016, 6, 1, 0, 0)),
    (datetime(2016, 6, 1, 0, 0), datetime(2017, 6, 1, 0, 0)),
    (datetime(2017, 6, 1, 0, 0), datetime(2018, 6, 1, 0, 0)),
    (datetime(2018, 6, 1, 0, 0), datetime(2019, 6, 1, 0, 0)),
    (datetime(2019, 6, 1, 0, 0), datetime(2020, 6, 1, 0, 0)),
    (datetime(2020, 6, 1, 0, 0), datetime(2021, 6, 1, 0, 0)),
)

# %%
delay_to_df[180]

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

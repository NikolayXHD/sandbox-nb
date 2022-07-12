from __future__ import annotations

from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_DIR_PATTERN = 'lvl_?'
FEATHER_FILE_PATTERN = '*.feather'


def build_df(directory: Path, max_list_level: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    last_msg: str | None = None

    directory = directory.resolve()
    print(directory)

    level_subdirs = list(
        sorted(
            level_dir
            for level_dir in directory.glob(LEVEL_DIR_PATTERN)
            if _parse_level(level_dir) <= max_list_level
        )
    )
    tickers = list(
        sorted(
            _parse_ticker(f)
            for level_dir in level_subdirs
            for f in level_dir.glob(FEATHER_FILE_PATTERN)
        )
    )

    for level_dir in level_subdirs:
        level = _parse_level(level_dir)
        files = sorted(level_dir.glob(FEATHER_FILE_PATTERN))
        for i, f in enumerate(files):
            if last_msg is not None:
                print('\r' + ' ' * len(last_msg) + '\r', end='')
            last_msg = f'{f.name} {i + 1} / {len(files)}'
            print(last_msg, end='')

            df_specific_ticker = pd.read_feather(f)
            df_specific_ticker = df_specific_ticker.assign(
                **{
                    'ticker': pd.Categorical(
                        [_parse_ticker(f)] * len(df_specific_ticker),
                        categories=tickers,
                    ),
                    'level': pd.Series(
                        [level] * len(df_specific_ticker),
                        dtype=np.byte,
                    ),
                }
            )
            parts.append(df_specific_ticker)
    print()
    df = pd.concat(parts, ignore_index=True)
    df.sort_values(by=['t'], inplace=True, ignore_index=True)
    df = df.assign(**{'w': get_w(df)})
    return df


def _parse_level(lvl_subdir: Path) -> int:
    return int(lvl_subdir.stem[-1])


def _parse_ticker(file_ticker: Path) -> str:
    return file_ticker.stem


def get_w(df: pd.DataFrame) -> pd.Series:
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

    df_agg_day_ticker = (
        df_agg_source.groupby(['d', 'ticker'])
        .agg(**{'n_td': ('t', 'count')})
        .reset_index()
    )

    df_agg_day_ticker = df_agg_day_ticker.assign(
        **{'ln_n_td': np.log(1 + df_agg_day_ticker['n_td'])}
    )

    df_agg_day = (
        df_agg_day_ticker.groupby('d')
        .agg(**{'sum_t_ln_n_td': ('ln_n_td', 'sum')})
        .reset_index()
    )

    df_agg_result = df_agg_source.merge(df_agg_day, on='d', copy=False).merge(
        df_agg_day_ticker, on=('d', 'ticker'), copy=False
    )

    return (
        df_agg_result['ln_n_td']
        / df_agg_result['n_td']
        / df_agg_result['sum_t_ln_n_td']
    )


def filter_df_by_dates(
    df: pd.DataFrame, date_from: datetime | None, date_to: datetime | None
) -> pd.DataFrame:
    """
    Assume time variable is sorted, binary search boundary indices, slice
    by indices thus avoiding copy.

    date_from: inclusive left boundary
    date_to: exclusive right boundary
    """
    index_t_from = (
        np.searchsorted(df['t'], date_from.timestamp(), side='right')
        if date_from is not None
        else 0
    )
    index_t_to = (
        np.searchsorted(df['t'], date_to.timestamp(), side='right')
        if date_to is not None
        else len(df)
    )
    return df.iloc[index_t_from:index_t_to]


__all__ = ['build_df', 'filter_df_by_dates']

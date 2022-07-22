from __future__ import annotations

from datetime import datetime
import itertools
from pathlib import Path
import typing

import numpy as np
import pandas as pd

from .balance import get_w


def build_df(
    directory: Path,
    max_list_level: int,
    fields: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    last_msg: str | None = None

    directory = directory.resolve()
    print(directory)

    level_to_tickers = {
        level: sorted(
            set(
                _parse_ticker(f)
                for f in (directory / 't' / f'lvl_{level}').glob('*.feather')
            )
        )
        for level in range(1, max_list_level + 1)
    }
    tickers = sorted(set(itertools.chain(*level_to_tickers.values())))
    time_field_requested = 't' in fields
    if not time_field_requested:
        # time is required for sorting
        fields.insert(0, 't')

    for lvl, tickers in level_to_tickers.items():
        for i, ticker in enumerate(tickers):
            if last_msg is not None:
                print('\r' + ' ' * len(last_msg) + '\r', end='')
            last_msg = f'lvl_{lvl} {ticker} {i + 1} / {len(tickers)}'
            print(last_msg, end='')
            df = pd.DataFrame()
            for field in fields:
                if field == 'w':
                    continue
                elif field == 'ticker':
                    ds = pd.Categorical(
                        [ticker] * len(df), categories=tickers,
                    )
                elif field == 'level':
                    ds = pd.Series(
                        [lvl] * len(df), dtype=np.byte
                    )
                else:
                    df_field = pd.read_feather(
                        directory / field / f'lvl_{lvl}' / f'{ticker}.feather'
                    )
                    ds = df_field.loc[:, field]
                if df.shape[1] > 0:
                    assert len(ds) == len(df)
                df.loc[:, field] = ds
            parts.append(df)
    print()
    df_all_tickers = pd.concat(parts, ignore_index=True)
    df_all_tickers.sort_values(by=['t'], inplace=True, ignore_index=True)
    if 'w' in fields:
        assert 'ticker' in fields
        df_all_tickers.loc[:, 'w'] = get_w(
            t=df_all_tickers['t'], ticker=df_all_tickers['ticker']
        )
    if not time_field_requested:
        df_all_tickers.drop('t', axis=1)

    return df_all_tickers


def _parse_level(lvl_subdir: Path) -> int:
    return int(lvl_subdir.stem[-1])


def _parse_ticker(file_ticker: Path) -> str:
    return file_ticker.stem


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
    assert isinstance(index_t_from, typing.SupportsInt), str(index_t_from)
    assert isinstance(index_t_to, typing.SupportsInt), str(index_t_to)
    return df.iloc[int(index_t_from): int(index_t_to)]


__all__ = ['build_df', 'filter_df_by_dates', 'get_w']

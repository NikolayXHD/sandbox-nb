from __future__ import annotations

from datetime import datetime
import typing

from pathlib import Path

import numpy as np
import pandas as pd

from .balance import get_w

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
    df = df.assign(
        **{'w': get_w(t=df['t'], ticker=df['ticker'])}
    )
    return df


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

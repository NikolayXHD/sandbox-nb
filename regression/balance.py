from __future__ import annotations

import numpy as np
import pandas as pd


def get_w(*, ticker: pd.Series, t: pd.Series) -> pd.Series:
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
    df_agg_source = pd.DataFrame({ticker.name: ticker, 'd': t // (3600 * 24)})

    df_agg_day_ticker = df_agg_source.groupby(['d', ticker.name]).agg(
        **{'n_td': (ticker.name, 'count')}
    ).reset_index()

    df_agg_day_ticker = df_agg_day_ticker.assign(
        **{'ln_n_td': np.log1p(df_agg_day_ticker['n_td'])}
    )

    df_agg_day = df_agg_day_ticker.groupby('d').agg(
        **{'sum_t_ln_n_td': ('ln_n_td', 'sum')}
    ).reset_index()

    df_agg_result = df_agg_source.merge(df_agg_day, on='d', copy=False).merge(
        df_agg_day_ticker, on=('d', ticker.name), copy=False
    )

    return (
        df_agg_result['ln_n_td']
        / df_agg_result['n_td']
        / df_agg_result['sum_t_ln_n_td']
    )


def get_weight_total_per_day(*, t: pd.Series, w: pd.Series) -> pd.Series:
    """
    For given time and weight series, return series of same size where each
    element is total weight per day
    """
    return (
        pd.DataFrame({'d': t // (3600 * 24), w.name: w}).groupby('d')
        .transform('sum')[w.name]
    )


__all__ = ['get_w', 'get_weight_total_per_day']

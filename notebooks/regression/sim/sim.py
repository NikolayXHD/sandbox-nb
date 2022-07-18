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

# %% tags=[]
from __future__ import annotations

from array import array
from collections import defaultdict
import typing

import numpy as np
import pandas as pd

from regression.balance import get_weight_total_per_day
from regression.load import filter_df_by_dates


def sim(
    get_score: typing.Callable[[pd.DataFrame], np.ndarray],
    *,
    profit_fld: str = 'profit_in_currency',
    use_validation_df: bool = False,
    title: str | None = None,
) -> None:
    df = sim_report_df(
        get_score,
        profit_fld=profit_fld,
        use_validation_df=use_validation_df,
        title=title,
    )
    df_txt = df.assign(
        **{
            'freq': df['freq'].map('{:.4f}'.format),
            'prof': df['prof'].map('{:+4_.2f}'.format),
        }
    )
    mask = df['type'] == 'rnd'
    df_txt.loc[mask, ['prof']] = df_txt.loc[mask, ['prof']].applymap(
        lambda txt:'| ' + txt
    )

    df_pivot = df_txt.pivot(
        index=['period', 'freq'], columns=['d', 'type']
    ).droplevel(0, axis=1)

    print (title)
    with pd.option_context('display.width', 120):
        print(df_pivot)
        print()


def sim_report_df(
    get_score: typing.Callable[[pd.DataFrame], np.ndarray],
    *,
    profit_fld: str = 'profit_in_currency',
    use_validation_df: bool = False,
    title: str | None = None,
) -> pd.DataFrame:
    periods: list[str] = []
    delays: list[int] = []
    freqs: list[float | None] = []
    profits: list[float | None] = []
    profit_types: list[str] = []

    period_to_freq: dict[str, float] = {}

    for delay_ix, delay in enumerate(delay_to_df.keys()):
        df_full = get_df(
            delay=delay,
            date_from=None,
            date_to=None,
            use_validation_df=use_validation_df,
        )
        s_score_full = get_score(df_full)
        df_full = df_full.assign(
            **{
                'score': s_score_full,
                'w_day_total': get_weight_total_per_day(
                    t=df_full['t'], w=(s_score_full > 0) * df_full['w']
                )
            }
        )
        for date_from, date_to in iterate_date_ranges(
            append_empty_range=True, use_validation_df=use_validation_df
        ):
            period = f'{format_date(date_from)} -'
            df = filter_df_by_dates(df_full, date_from, date_to)
            s_score = df['score']
            s_profits = df[profit_fld]
            w_day_total = df['w_day_total']
            s_w = df['w']
            if delay_ix == 0:
                # because the data has extra year for validation,
                # there should be no differences in freq, neutral_verage_w
                # due to differences in right boundary truncation for different
                # profit delays
                freq = (s_score > 0).sum() / len(df)
                period_to_freq[period] = freq
            else:
                freq = period_to_freq[period]

            for profit_ix, profit_type in enumerate(('rnd', 'avg', 'adj')):
                periods.append(period)
                delays.append(delay)
                freqs.append(freq)
                profit_types.append(profit_type)

                match profit_type:
                    case 'adj':
                        try:
                            positive_average_w = np.average(
                                s_profits,
                                weights=(
                                    s_score * s_w / w_day_total.replace(0, 1)
                                )
                            )
                        except ZeroDivisionError:
                            positive_average_w = None
                        profits.append(positive_average_w)
                    case 'avg':
                        try:
                            positive_average_w = np.average(
                                s_profits, weights=s_score * s_w
                            )
                        except ZeroDivisionError:
                            positive_average_w = None
                        profits.append(positive_average_w)
                    case 'rnd':
                        neutral_average_w = np.average(s_profits, weights=s_w)
                        profits.append(neutral_average_w)
                    case _:
                        assert False, f'Unknown profit type {profit_type}'

    df = pd.DataFrame(
        {
            'period': pd.Series(periods),
            'd': pd.Series(delays),
            'freq': pd.Series(freqs),
            'prof': pd.Series(profits),
            'type': pd.Series(profit_types)
        }
    )
    return df


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Example of treating desirability as probability multiplier
#
# Since we buy more stock with hihger score, it's logical these stocks will
# have bigger impact on profits.
#
# However, simply multiplying existing "probablity" weight by "desirability"
# score, ignores the practical limitation.
#
# When the day comes and extreme score is observed, our possibilities to buy
# more are limited by our total porfolio value.
#
# We cannot borrow money from the past (time machine needed), nor we want to
# borrow from the future (we hate banks).
#
# Therefore relative desirability can only rebalance weights between the stocks
# available at the same time. (Not implemented yet)

# %% tags=[] jupyter={"outputs_hidden": true}
def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
    x = df['dlnv_3d']
    y = df['dln_24d']
    slope_bin = 4 / 8
    bin_x = 0.025
    bin_y = 0.01
    min_x = -0.1
    min_y = -0.03 + bin_y * 0.85
    score = np.maximum(
        0, slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y
    )
    return score


sim(_get_score)

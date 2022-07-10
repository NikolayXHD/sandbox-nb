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

# %%
import typing

import numpy as np
import pandas as pd


def sim(
    get_score: typing.Callable[[pd.DataFrame], np.array],
    profit_fld: str = 'profit_in_currency',
    use_validation_df: bool = False,
) -> None:
    for delay in delay_to_df.keys():
        print(delay)
        for date_from, date_to in iterate_date_ranges(
            append_empty_range=True, use_validation_df=use_validation_df
        ):
            print(f'    {format_date(date_from)} -- {format_date(date_to)}')

            df = get_df(
                delay=delay,
                date_from=date_from,
                date_to=date_to,
                use_validation_df=use_validation_df,
            )

            score = get_score(df)

            positive_average = np.average(df[profit_fld].values, weights=score)
            positive_average_w = np.average(
                df[profit_fld].values, weights=score * df['w']
            )
            neutral_average = np.average(df[profit_fld].values)
            neutral_average_w = np.average(df[profit_fld].values, weights=df['w'])

            print('            ~w     w      freq')
            print(
                f'    score:  {positive_average:+4_.2f}  {positive_average_w:+4_.2f}  '
                f'{(score > 0).sum() / len(df):.4f}'
            )
            print(
                f'      all:  {neutral_average:+4_.2f}  {neutral_average_w:+4_.2f}'
            )
            print()


# %% jupyter={"outputs_hidden": true} tags=[]
def _get_score(df: pd.DataFrame) -> np.array:
    indicator_1_fld = 'dln_exp_3d'
    indicator_2_fld = 'dln_exp_no_vol_24d'

    slope_bin = 4 / 8
    bin_x = 0.025
    bin_y = 0.01
    min_x = -0.1
    min_y = -0.03 + bin_y * 0.85

    x = df[indicator_1_fld]
    y = df[indicator_2_fld]
    score = np.maximum(
        0, slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y
    )
    return score

sim(_get_score, use_validation_df=True)


# %% tags=[] jupyter={"outputs_hidden": true}
def _get_score(df: pd.DataFrame) -> np.array:
    indicator_1_fld = 'dln_exp_3d'
    indicator_2_fld = 'dln_exp_no_vol_24d'

    slope_bin = 4 / 8
    bin_x = 0.025
    bin_y = 0.01
    min_x = -0.1
    min_y = -0.03 + bin_y * 0.85

    x = df[indicator_1_fld]
    y = df[indicator_2_fld]
    score = np.maximum(
        0, slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y
    )
    return score

sim(_get_score)


# %% jupyter={"outputs_hidden": true} tags=[]
def _get_score(df: pd.DataFrame) -> np.array:
    return (
        df['dln_exp_4h'].between(-1, +1)
        & df['dln_exp_no_vol_24d'].between(-1, -0.008)
    ).astype('int')

sim(_get_score)


# %% jupyter={"outputs_hidden": true} tags=[]
def _get_score(df: pd.DataFrame) -> np.array:
    indicator_1_fld = 'dln_exp_24d'
    indicator_2_fld = 'dln_exp_no_vol_24d'

    slope_bin = 0
    bin_x = 0.025
    bin_y = 0.01
    min_x = -0.1
    min_y = -0.03 + bin_y * 2.5

    x = df[indicator_1_fld]
    y = df[indicator_2_fld]
    score = np.maximum(
        0, slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y
    )
    return score

sim(_get_score)

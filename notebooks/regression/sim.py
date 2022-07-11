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
from collections import defaultdict
import typing

import numpy as np
import pandas as pd


def sim(
    get_score: typing.Callable[[pd.DataFrame], np.array],
    *,
    profit_fld: str = 'profit_in_currency',
    use_validation_df: bool = False,
    title: str | None = None,
) -> None:
    col_to_lines = defaultdict(list)

    print_column('', 0, col_to_lines)
    print_column(title or '', 0, col_to_lines)
    print_column('------------------------', 0, col_to_lines)
    
    for date_from, date_to in iterate_date_ranges(
        append_empty_range=True, use_validation_df=use_validation_df
    ):
        if date_from is None:
            print_column('------------------------', 0, col_to_lines)
        print_column(
            f'{format_date(date_from)} -- {format_date(date_to)}',
            0,
            col_to_lines,
        )

    print_column('', 1, col_to_lines)
    print_column(f'freq', 1, col_to_lines)
    print_column('-----', 1, col_to_lines)    

    for column_ix, delay in enumerate(delay_to_df.keys()):
        print_column(str(delay), 2 + column_ix, col_to_lines)
        # print_column('~w/rnd ~w     +w/rnd  +w    ', 2 + column_ix, col_to_lines)
        # print_column('--------------------------', 2 + column_ix, col_to_lines)
        print_column('+w/rnd  +w    ', 2 + column_ix, col_to_lines)
        print_column('------------', 2 + column_ix, col_to_lines)

    for date_from, date_to in iterate_date_ranges(
        append_empty_range=True, use_validation_df=use_validation_df
    ):
        for column_ix, delay in enumerate(delay_to_df.keys()):
            df = get_df(
                delay=delay,
                date_from=date_from,
                date_to=date_to,
                use_validation_df=use_validation_df,
            )

            score = get_score(df)

            # try:
            #     positive_average = np.average(df[profit_fld].values, weights=score)
            # except ZeroDivisionError:
            #     positive_average = np.nan

            try:
                positive_average_w = np.average(
                    df[profit_fld].values, weights=score * df['w']
                )
            except ZeroDivisionError:
                positive_average_w = np.nan

            # neutral_average = np.average(df[profit_fld].values)
            neutral_average_w = np.average(df[profit_fld].values, weights=df['w'])

            if column_ix == 0:
                if date_from is None:
                    print_column('----', 1 + column_ix, col_to_lines)
                print_column(f'{(score > 0).sum() / len(df):.4f}', 1 + column_ix, col_to_lines)

            if date_from is None:
                print_column('------------', 2 + column_ix, col_to_lines)
            print_column(
                (
                    # f'{neutral_average:+4_.2f}  {positive_average:+4_.2f}  '
                    f'{neutral_average_w:+4_.2f}  {positive_average_w:+4_.2f}'
                ),
                2 + column_ix,
                col_to_lines,
            )

    i_to_width = {
        i: max(len(l) for l in lines) for i, lines in col_to_lines.items()
    }
    num_lines = max(len(lines) for lines in col_to_lines.values())
    text = '\n'.join(
        ' | '.join(
            (
                col_to_lines[column_ix][line_ix]
                if line_ix < len(col_to_lines[column_ix])
                else ''
            ).ljust(i_to_width[column_ix])
            for column_ix in range(len(col_to_lines))
        )
        for line_ix in range(num_lines)
    )
    print(text)


def print_column(txt: str, i: int, col_to_lines) -> None:
    col_to_lines[i].append(txt)    


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

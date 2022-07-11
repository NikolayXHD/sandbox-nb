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


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8  #
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = (
        ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(-0.45, 0.45)
        * x.between(-0.50, 0.50).astype('int')
    )
    return score

sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8  #
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = (
        ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(-0.40, 0.30)
        * x.between(-0.50, 0.50).astype('int')
    )
    return score

sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8  #
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = (
        ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(-0.30, 0.25)
        * x.between(-0.50, 0.50).astype('int')
    )
    return score

sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = (
        ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(-0.30, 0.15)
        * x.between(-0.50, 0.50).astype('int')
    )
    return score

sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %%
def sim_ranges(val_min, val_max, step):
    for val in np.arange(val_min, val_max, step):
        def _get_score(df: pd.DataFrame) -> np.array:
            x = df['dln_exp_no_vol_log_3d']
            y = df['dln_exp_no_vol_log_24d']
            slope_bin = 0.5 / 8
            bin_x = 0.2
            bin_y = 0.2
            min_x = -0.8
            score =(
                (y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(val, val + step)
                # ~y.between(-0.30, 0.30)
                & x.between(-0.50, 0.50)
            )
            return score
        sim(_get_score, title=f'{val:.2f} -- {val + step:.2f}')
        print()

sim_ranges(-0.8, 0.8, 0.1)


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score =(
        ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(-0.40, 1)
        # ~y.between(-0.30, 0.35)
        & x.between(-0.50, 0.50)
    )
    return score

sim(_get_score)


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 1 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    min_y = +0.25
    score =(
        ((y - min_y) / bin_y > slope_bin * (x - min_x) / bin_x)
        & x.between(-0.5, 0.5)
    )
    return score

sim(_get_score)


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 1 / 5
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.5
    min_y = 0.15
    score = slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y < 0
    return score

sim(_get_score)


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 1 / 5
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.5
    min_y = -0.4
    score = slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y > 0
    return score

sim(_get_score)


# %% tags=[] jupyter={"outputs_hidden": true}
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_3d']
    y = df['dln_exp_no_vol_24d']
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


# %%
def _get_score(df: pd.DataFrame) -> np.array:
    x = df['dln_exp_3d']
    y = df['dln_exp_no_vol_24d']
    return (
        x.between(-0.07, +0.05).astype('int') *
        np.maximum(0, ((x / 0.01) ** 2 + (y / (0.0025)) ** 2) ** 0.5 - 1.5)
    )
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

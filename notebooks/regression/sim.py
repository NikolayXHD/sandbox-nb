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

# %%
from __future__ import annotations

from collections import defaultdict
import typing

import numpy as np
import pandas as pd


def sim(
    get_score: typing.Callable[[pd.DataFrame], np.ndarray],
    *,
    profit_fld: str = 'profit_in_currency',
    use_validation_df: bool = False,
    title: str | None = None,
) -> None:
    col_to_lines: defaultdict[int, list[str]] = defaultdict(list)

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
    print_column('freq', 1, col_to_lines)
    print_column('-----', 1, col_to_lines)

    for column_ix, delay in enumerate(delay_to_df.keys()):
        print_column(str(delay), 2 + column_ix, col_to_lines)
        # print_column(
        #     '~w/rnd ~w     +w/rnd  +w    ', 2 + column_ix, col_to_lines
        # )
        # print_column(
        #     '--------------------------', 2 + column_ix, col_to_lines
        # )
        print_column('+w/rnd  +w  ', 2 + column_ix, col_to_lines)
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
            #     positive_average = np.average(
            #         df[profit_fld].values, weights=score
            #     )
            # except ZeroDivisionError:
            #     positive_average = np.nan

            try:
                positive_average_w = np.average(
                    df[profit_fld].values, weights=score * df['w']
                )
            except ZeroDivisionError:
                positive_average_w = np.nan

            # neutral_average = np.average(df[profit_fld].values)
            neutral_average_w = np.average(
                df[profit_fld].values, weights=df['w']
            )

            if column_ix == 0:
                if date_from is None:
                    print_column('----', 1 + column_ix, col_to_lines)
                print_column(
                    f'{(score > 0).sum() / len(df):.4f}',
                    1 + column_ix,
                    col_to_lines,
                )

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


# %% [markdown]
# ## Validate high profit indicator ranges on 2022 stock history
#
# it confidently repeats

# %% jupyter={"outputs_hidden": true} tags=[]
for min_val, max_val in (
    (-0.30, 0.15),
    (-0.30, 0.25),
    (-0.40, 0.30),
    (-0.45, 0.45),
):

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_exp_no_vol_log_3d']
        y = df['dln_exp_no_vol_log_24d']
        slope_bin = 0.5 / 8  #
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) * x.between(-0.50, 0.50).astype('int')
        return score

    sim(
        _get_score,
        title=f'{min_val:.2f} -- {max_val:.2f}',
        use_validation_df=True,
    )
    print()

# %% [markdown]
# ## Narrowing down of long indicator range `dln_exp_no_vol_log_24d`
#
# ```
# frequency       | 180d  30d   7d    |
# ----------------+-------------------+
# 0.16            | +0.37 +0.30 +0.22 |
# 0.05            | +0.75 +0.53 +0.40 |
# 0.016           | +1.51 +0.88 +0.56 |
# 0.006           | +2.21 +1.31 +0.57 |
# ----------------+-------------------+
# ```

# %% tags=[]
for min_val, max_val in (
    (-0.30, 0.15),
    (-0.30, 0.25),
    (-0.40, 0.30),
    (-0.45, 0.45),
):

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_exp_no_vol_log_3d']
        y = df['dln_exp_no_vol_log_24d']
        slope_bin = 0.5 / 8  #
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) * x.between(-0.50, 0.50).astype('int')
        return score

    sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')
    print()


# %% [markdown]
# ## Slice short indicator `dln_exp_no_vol_log_3d`

# %% tags=[]
def sim_ranges(val_min, val_max, step):  # type: ignore[no-redef]
    for val in np.arange(val_min, val_max, step):

        # type: ignore[no-redef]
        def _get_score(df: pd.DataFrame) -> np.ndarray:
            x = df['dln_exp_no_vol_log_3d']
            y = df['dln_exp_no_vol_log_24d']
            slope_bin = 0.5 / 8
            bin_x = 0.2
            bin_y = 0.2
            min_x = -0.8
            score = (
                (y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
                    val, val + step
                )
                # ~y.between(-0.30, 0.30)
                & x.between(-0.50, 0.50)
            )
            return score

        sim(_get_score, title=f'{val:.2f} -- {val + step:.2f}')
        print()


sim_ranges(-0.8, 0.8, 0.1)


# %% [markdown]
# ## Maximal 180d profit
#
# since extreme values of long indicator have high 180d income and low /
# negative 7d, 30d incomes, maximal 180d income is achieved at 4 corners

# %% tags=[]
def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
    x = df['dln_exp_no_vol_log_3d']
    y = df['dln_exp_no_vol_log_24d']
    slope_bin = 0.5 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
        -0.30, 0.15
    ) & ~x.between(-0.25, 0.50).astype('int')
    return score


sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %%

# %% [markdown]
# ## Slice long indicator `dln_exp_no_vol_log_24d`

# %% jupyter={"outputs_hidden": true} tags=[]
def sim_ranges(val_min, val_max, step):  # type: ignore[no-redef]
    for val in np.arange(val_min, val_max, step):

        # type: ignore[no-redef]
        def _get_score(df: pd.DataFrame) -> np.ndarray:
            x = df['dln_exp_no_vol_log_3d']
            y = df['dln_exp_no_vol_log_24d']
            slope_bin = 0.5 / 8
            bin_x = 0.2
            bin_y = 0.2
            min_x = -0.8
            score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
                -0.3, 0.2
            ) & x.between(val, val + step)
            return score

        sim(_get_score, title=f'{val:.2f} -- {val + step:.2f}')
        print()


sim_ranges(-0.9, 0.9, 0.1)


# %% [markdown]
# ## dln_exp_no_vol_log_72d
#
# higher frequency of high 30d profit
#
# ```
# frequency       | 180d  30d   7d    |
# ----------------+-------------------+
# 0.13            | +0.32 +0.47 +0.30 |
# 0.045           | +0.46 +0.72 +0.44 |
# 0.02            | +0.53 +0.92 +0.49 |
# ----------------+-------------------+
# ```

# %% tags=[]
def sim_range(min_val, max_val):  # type: ignore[no-redef]
    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_exp_no_vol_log_3d']
        y = df['dln_exp_no_vol_log_72d']
        slope_bin = 0.25 / 8
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) & x.between(-1, 0.50)
        return score

    sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')


for min_val, max_val in (
    (-0.30, +0.35),
    (-0.40, +0.40),
    (-0.45, +0.45),
):
    sim_range(min_val, max_val)
    print()


# %% [markdown]
# ## Example of treating desirability as probability multiplier
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

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

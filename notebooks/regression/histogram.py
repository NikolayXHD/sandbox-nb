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

import math

from matplotlib import colors
from matplotlib import pyplot as plt


def plot_2d_hist(
    df_k,
    n_days,
    adv_field,
    profit_field,
    ax=None,
    plot_xlabel=True,
    plot_ylabel=True,
    bins=(400, 400),
    val_range=None,
    plot_values=False,
    ignore_weight=False,
    log_color_scale=True,
):
    ax.grid(False)

    hist, xbins, ybins, im = ax.hist2d(
        df_k[adv_field],
        df_k[profit_field],
        weights=None if ignore_weight else df_k['w'],
        bins=bins,
        range=val_range,
        cmin=1 if ignore_weight else df_k['w'].min(),
        cmap='Blues',
        norm=colors.LogNorm() if log_color_scale else None,
    )

    if plot_values:
        x_delta = (xbins[-1] - xbins[0]) / len(xbins)
        y_delta = (ybins[-1] - ybins[0]) / len(ybins)
        for i in range(len(ybins) - 1):
            for j in range(len(xbins) - 1):
                if hist[j, i] > 0:
                    ax.text(
                        xbins[j] + x_delta / 2,
                        ybins[i] + y_delta / 2,
                        format_number(hist[j, i], 0),
                        color='w',
                        fontsize=14,
                        bbox={'alpha': 0.15, 'facecolor': 'b'},
                    )

    if plot_xlabel:
        ax.set_xlabel(adv_field)
    if plot_ylabel:
        ax.set_ylabel(profit_field + ' ' + str(n_days))
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)


def format_number(value, n) -> str:
    for num, qual in zip(
        (10 ** (3 * i) for i in range(4, -5, -1)),
        ('T', 'G', 'M', 'K', '', 'm', 'μ', 'n', 'p'),
    ):
        if value >= num:
            return f'{int(round(value / num, n))}{qual}'
    return str(value)


def plot_histogram_pairs(field_grps, **kwargs):
    profit_field = 'profit_in_currency'

    fig, axes = plt.subplots(
        figsize=(
            5 * len(delay_to_df) * len(field_grps[0]),
            5 * len(field_grps),
        ),
        ncols=len(delay_to_df) * len(field_grps[0]),
        nrows=len(field_grps),
    )
    plt.tight_layout()

    for i, fields in enumerate(field_grps):
        for j, field in enumerate(fields):
            for k, num_days in enumerate(delay_to_df.keys()):
                row = i
                col = j * len(delay_to_df) + k
                ax=axes[row, col]
                plot_2d_hist(
                    delay_to_df[num_days],
                    num_days,
                    field,
                    profit_field,
                    ax=ax,
                    bins=(100, 100),
                    **kwargs,
                )
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

    plt.show()


# %%
plot_histogram_pairs(
    [
        (f'dlnv_log_{duration}', f'dln_log_{duration}')
        for duration in durations
    ]
)

# %%
plot_histogram_pairs(
    [
        (f'dlnv_{duration}', f'dln_{duration}')
        for duration in durations
    ]
)

# %%
plot_histogram_pairs(
    [
        (f'adv_log_{duration}', f'ad_log_{duration}')
        for duration in durations
    ],
    log_color_scale=False,
)

# %% [markdown]
# `ad_3d` looks like accordion, probably due to weekends

# %%
plot_histogram_pairs(
    [(f'adv_{duration}', f'ad_{duration}') for duration in durations],
    log_color_scale=False,
)

# %% [markdown]
# ## Demonstrate ignore_weight difference

# %% tags=[] jupyter={"source_hidden": true}
fig, axes = plt.subplots(figsize=(12, 5), ncols=2)

for i, ignore_weight in enumerate((True, False)):
    plot_2d_hist(
        delay_to_df[7],
        7,
        'dlnv_3d',
        'dln_24d',
        ax=axes[i],
        bins=(8, 8),
        val_range=((-0.1, +0.1), (-0.04, +0.04)),
        plot_values=True,
        ignore_weight=ignore_weight,
    )
plt.show()

# %% tags=[] jupyter={"source_hidden": true}
fig, ax = plt.subplots(figsize=(5, 5))
plot_2d_hist(
    delay_to_df[7],
    7,
    'dlnv_24d',
    'dln_24d',
    ax=ax,
    bins=(8, 6),
    val_range=((-0.1, +0.1), (-0.030, +0.030)),
    plot_values=True,
)
plt.show()

# %% tags=[] jupyter={"source_hidden": true}
fig, ax = plt.subplots(figsize=(5, 5))
plot_2d_hist(
    delay_to_df[7],
    7,
    'dlnv_3d',
    'dln_72d',
    ax=ax,
    bins=(8, 9),
    val_range=((-0.1, +0.1), (-0.0125, +0.010)),
    plot_values=True,
)
plt.show()


# %% [markdown]
# # test `log_scale_value`

# %% tags=[] jupyter={"source_hidden": true}
def test_log_scale_value():
    delay = 7
    df = delay_to_df[delay]
    df_test = df[['dlnv_3d', 'dln_24d', 'w']].assign(
        **{
            'dlnv_log_3d': log_scale_value(df['dlnv_3d'], 10 ** 4),
            'dln_log_24d': log_scale_value(df['dln_24d'], 10 ** 4),
        }
    )

    fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
    for i, (field, log_field) in enumerate(
        (
            ('dlnv_log_3d', 'dlnv_3d'),
            ('dln_log_24d', 'dln_24d'),
        )
    ):
        plot_2d_hist(
            df_test,
            delay,
            log_field,
            field,
            ax=axes[i],
            bins=(100, 100),
            plot_values=False,
            log_color_scale=False,
        )

    plt.show()


test_log_scale_value()

# %% tags=[] jupyter={"source_hidden": true}
fig, ax = plt.subplots(figsize=(5, 5))

plot_2d_hist(
    delay_to_df[7],
    7,
    'ad_72d',
    'dln_log_72d',
    ax=ax,
    bins=(100, 100),
    plot_values=False,
    log_color_scale=True,
)

plt.show()

# %% tags=[] jupyter={"source_hidden": true}
# # %%timeit -n1 -r1
# 37.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

import numpy as np
import pandas as pd
import seaborn as sns


def plot_histograms(*, adv_field: str, profit_field: str) -> None:
    delay_to_Xy_1d = {
        delay: separate_features_1d(
            delay=delay,
            dt_from=None,
            dt_to=None,
            adv_field=adv_field,
            profit_field=profit_field,
        )
        for delay, df in delay_to_df.items()
    }

    fig, axes = plt.subplots(1, len(delay_to_Xy_1d), figsize=(24, 7))
    fig.suptitle('Дискретизированные значения')

    bins = (50, 50)

    for i, (num_days, (X, y, w)) in enumerate(delay_to_Xy_1d.items()):
        # noinspection PyProtectedMember
        X_d, y_d, sample_weight = histogram.Histogram2dRegressionWrapper(
            None,
            bins,
            memory_,
            verbose=False,
        )._get_histogram(X, y, None, bins, same_x=False)

        df_d = pd.DataFrame(
            {
                'indicator': pd.Series(np.ravel(X_d)),
                'profit': pd.Series(y_d),
                'weights': pd.Series(sample_weight),
            }
        )
        plt.subplot(1, 3, i + 1)
        ax = sns.histplot(
            ax=axes[i],
            data=df_d,
            x='indicator',
            y='profit',
            weights='weights',
            bins=(200, 200),
            # bins=(50, 50),
        )
        ax.grid(True)
        ax.set_title(f'доход через {num_days} дней, в пересчёте на 1 год')
    plt.show()


plot_histograms(adv_field='indicator', profit_field='profit')

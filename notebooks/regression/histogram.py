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

import math

from matplotlib import colors
from matplotlib import pyplot as plt


def plot_2d_hist(
    df_k,
    n_days,
    indicator_field,
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
        df_k[indicator_field],
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
                        bbox={'alpha': 0.25, 'facecolor': 'b'},
                        # fontweight="bold",
                    )

    if plot_xlabel:
        ax.set_xlabel(indicator_field)
    if plot_ylabel:
        ax.set_ylabel(profit_field + ' ' + str(n_days))
    ax.tick_params(axis='x', direction='in', pad=-12)
    ax.tick_params(axis='y', direction='in', pad=-22)
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


# %%
fig, ax = plt.subplots(figsize=(15, 15))

plot_2d_hist(
    delay_to_df[7],
    7,
    'ad_exp_72d',
    'dln_exp_no_vol_log_72d',
    ax=ax,
    bins=(100, 100),
    plot_values=False,
    log_color_scale=True,
)

plt.show()

# %%
for delay, df in delay_to_df.items():
    df['dln_exp_log_3d'] = log_scale_value(df['dln_exp_3d'], 2000)
    df['dln_exp_no_vol_log_24d'] = log_scale_value(
        df['dln_exp_no_vol_24d'], 2000
    )

fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
plot_2d_hist(
    delay_to_df[7],
    7,
    'dln_exp_log_3d',
    'dln_exp_3d',
    ax=axes[0],
    bins=(100, 100),
    plot_values=False,
    log_color_scale=False,
)
plot_2d_hist(
    delay_to_df[7],
    7,
    'dln_exp_no_vol_log_24d',
    'dln_exp_no_vol_24d',
    ax=axes[1],
    bins=(100, 100),
    plot_values=False,
    log_color_scale=False,
)

plt.show()

# %% [markdown]
# ## Demonstrate ignore_weight difference

# %%
fig, axes = plt.subplots(figsize=(20, 10), ncols=2)

for i, ignore_weight in enumerate((True, False)):
    plot_2d_hist(
        delay_to_df[7],
        7,
        'dln_exp_3d',
        'dln_exp_no_vol_24d',
        ax=axes[i],
        bins=(8, 8),
        val_range=((-0.1, +0.1), (-0.04, +0.04)),
        plot_values=True,
        ignore_weight=ignore_weight,
    )
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))
plot_2d_hist(
    delay_to_df[7],
    7,
    'dln_exp_24d',
    'dln_exp_no_vol_24d',
    ax=ax,
    bins=(8, 6),
    val_range=((-0.1, +0.1), (-0.030, +0.030)),
    plot_values=True,
)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))
plot_2d_hist(
    delay_to_df[7],
    7,
    'dln_exp_3d',
    'dln_exp_no_vol_72d',
    ax=ax,
    bins=(8, 9),
    val_range=((-0.1, +0.1), (-0.0125, +0.010)),
    plot_values=True,
)
plt.show()


# %%
def plot_histogram_pairs(field_grps):
    profit_field = 'profit_in_currency'

    fig, axes = plt.subplots(
        figsize=(
            15 * len(field_grps[0]),
            5 * len(delay_to_df) * len(field_grps),
        ),
        nrows=len(delay_to_df) * len(field_grps),
        ncols=len(field_grps[0]),
    )
    plt.subplots_adjust(wspace=0.01, hspace=0.08)

    for i, num_days in enumerate(delay_to_df.keys()):
        for j, fields in enumerate(field_grps):
            for k, field in enumerate(fields):
                row = j * len(delay_to_df) + i
                plot_2d_hist(
                    delay_to_df[num_days],
                    num_days,
                    field,
                    profit_field,
                    ax=axes[row, k],
                    plot_ylabel=k > 0,
                    bins=(400, 200),
                )

    plt.show()


# %%
plot_histogram_pairs(
    [
        (f'dln_exp_log_{duration}', f'dln_exp_no_vol_log_{duration}')
        for duration in durations
    ]
)

# %%
plot_histogram_pairs(
    [
        (f'dln_exp_{duration}', f'dln_exp_no_vol_{duration}')
        for duration in durations
    ]
)

# %%
plot_histogram_pairs(
    [
        (f'indicator_log_{duration}', f'ad_exp_log_{duration}')
        for duration in durations
    ]
)

# %%
plot_histogram_pairs(
    [(f'indicator_{duration}', f'ad_exp_{duration}') for duration in durations]
)

# %%
# # %%timeit -n1 -r1
# 37.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

import numpy as np
import pandas as pd
import seaborn as sns


def plot_histograms(*, indicator_field: str, profit_field: str) -> None:
    delay_to_Xy_1d = {
        delay: separate_features_1d(
            delay=delay,
            dt_from=None,
            dt_to=None,
            indicator_field=indicator_field,
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


plot_histograms(indicator_field='indicator', profit_field='profit')

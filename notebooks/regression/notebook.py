# -*- coding: utf-8 -*-
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

# %% cellId="r60bke4u6ugchb8rkiobw"
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from joblib.memory import Memory
from sklearn import model_selection

from notebooks.regression import cross_validation
from notebooks.regression import histogram


def build_df(directory: Path):
    parts: list[pd.DataFrame] = []
    last_msg: str | None = None
    files = sorted(directory.glob('*.feather'))
    print(directory.resolve())
    for i, file in enumerate(files):
        if last_msg is not None:
            print('\r' + ' ' * len(last_msg) + '\r', end='')
        last_msg = f'{file.name} {i + 1} / {len(files)}'
        print(last_msg, end='')
        parts.append(pd.read_feather(file))
    print()
    df = pd.concat(parts)
    df.sort_values(by=['t'], inplace=True)
    return df


delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

PWD = Path(os.path.dirname(os.path.realpath('__file__')))
CACHE_STORAGE_PATH = PWD.joinpath('..', '..', '.storage', 'cache')
OUTPUT_STORAGE_PATH = PWD.joinpath(
    '..', '..', '..', 'sandbox', '.storage', 'output'
)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression-multiple-indicator',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2021-07-01',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

delay_to_df = {delay: build_df(path) for delay, path in delay_to_dir.items()}

# %%
delay_to_df[180]


# %% tags=[]
def plot_2d_hist(df_k, n_days, indicator_field, profit_field):
    x_field = indicator_field
    y_field = profit_field
    h, x_edges, y_edges = np.histogram2d(
        df_k[x_field], df_k[y_field], bins=(800, 800)
    )

    fig, ax = plt.subplots()
    ax.figure.set_figwidth(14)
    ax.figure.set_figheight(14)
    ax.set_title(f'доход через {n_days} дней, в пересчёте на 1 год')

    v_max_color = h.max()
    v_min_color = h[h > 0].min()
    v_num_color = 255

    ax.contourf(
        h,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        levels=np.linspace(
            v_min_color,
            v_max_color,
            num=v_num_color,
        ),
        cmap='Blues',
    )

    plt.show()


plot_2d_hist(delay_to_df[180], 180, 'indicator_12h', 'profit')
# plot_2d_hist(delay_to_df[180], 180, 'indicator_72d', 'profit_in_currency')

# %%
from notebooks.regression.memory import control_output


def separate_features_1d(
    *, delay, dt_from, dt_to, indicator_field, profit_field
):
    return (
        _separate_X_1d(
            delay=delay,
            dt_from=dt_from,
            dt_to=dt_to,
            indicator_field=indicator_field,
        ),
        _separate_y_1d(
            delay=delay,
            dt_from=dt_from,
            dt_to=dt_to,
            profit_field=profit_field,
        ),
    )


@control_output
@memory_.cache
def _mask_1d(*, delay, dt_from, dt_to):
    df_i = delay_to_df[delay]
    if dt_from is not None and dt_to is not None:
        return df_i['t'].between(dt_from.timestamp(), dt_to.timestamp())
    elif dt_from is not None:
        return df_i['t'] >= dt_from.timestamp()
    elif dt_to is not None:
        return df_i['t'] <= dt_to.timestamp()
    else:
        return None


@control_output
@memory_.cache
def _separate_X_1d(*, delay, dt_from, dt_to, indicator_field):
    df = delay_to_df[delay]
    result = df[[indicator_field]]
    mask = _mask_1d(delay=delay, dt_from=dt_from, dt_to=dt_to)
    if mask is not None:
        result = result[mask]
    return result.values


@control_output
@memory_.cache
def _separate_y_1d(*, delay, dt_from, dt_to, profit_field):
    df = delay_to_df[delay]
    result = df[profit_field]
    mask = _mask_1d(delay=delay, dt_from=dt_from, dt_to=dt_to)
    if mask is not None:
        result = result[mask]
    return result.values


# %%
import itertools


def build_df_indicator_quantiles(indicator_names: list[str]) -> pd.DataFrame:
    values = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    values_all = list(
        itertools.chain(values, (1 - v for v in reversed(values)))
    )
    indicator_name_to_q_values = {
        indicator_name: delay_to_df[180][indicator_name].quantile(values_all)
        for indicator_name in indicator_names
    }
    df = pd.DataFrame(
        {
            f'q_{value}': pd.Series(
                indicator_name_to_q_values[indicator_name].iloc[value_index]
                for indicator_name in indicator_names
            )
            for value_index, value in enumerate(values_all)
        }
    )
    df.index = pd.Index(indicator_names)
    return df


df_indicator_quantiles = build_df_indicator_quantiles(
    [
        'indicator_4h',
        'indicator_12h',
        'indicator_1d',
        'indicator_5d',
        'indicator',
        'indicator_72d',
        'market_indicator',
    ]
)

df_indicator_quantiles

# %%
from datetime import datetime
import typing

from matplotlib import ticker
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from sklearn import ensemble

CV = False
CV_USES_HISTOGRAM = True
CACHE = True


def create_estimator_bins_1d(delay):
    radius = 0.10

    def _w(d):
        return norm.pdf(d / radius)

    return histogram.Histogram2dRegressionWrapper(
        bins=(40, 1),
        shuffle=False,
        memory_=memory_,
        verbose=False,
        cache=CACHE,
        regressor=KNeighborsWeightedRegressor(
            n_neighbors=16,
            weights=_w,
            n_jobs=-1,
        ),
    )


def get_cv_1d():
    if CV_USES_HISTOGRAM:
        return cross_validation.CrossValidation(
            test_histogram_bins=(80, 20),
            verbose=False,
            cache=CACHE,
            memory_=memory_,
        )
    else:
        return cross_validation.CrossValidation(
            verbose=False,
            cache=CACHE,
            memory_=memory_,
        )


def get_label_1d(num_days, scores):
    return (
        f'{num_days:<3} days'
        # f'$R^2_{{oos}}$ {scores.mean(): .5f} ± {scores.std():.5f}'
    )


def plot_model_1d(
    reg_k,
    ax,
    *args,
    min_x=-1,
    max_x=+1,
    relative: float | None = None,
    **kwargs,
):
    num_points = 1000
    X_pred = np.linspace(min_x, max_x, num=num_points)
    y_pred = reg_k.predict(X_pred.reshape(-1, 1))
    if relative is not None:
        assert isinstance(relative, typing.SupportsFloat)
        y_relative = reg_k.predict(np.array([[relative]]))
        y_pred -= y_relative[0]
    ax.plot(X_pred, y_pred, *args, **kwargs)
    ax.grid(True)


def plot_regressions_1d(
    *,
    dt_from,
    dt_to,
    indicator_field,
    profit_field,
    axes,
    num_color=None,
    relative: float | None = None,
    y_ticks_interval: float | None = None,
    **kwargs,
):
    delay_to_Xy_1d = {
        delay: separate_features_1d(
            delay=delay,
            dt_from=dt_from,
            dt_to=dt_to,
            indicator_field=indicator_field,
            profit_field=profit_field,
        )
        for delay, df in delay_to_df.items()
    }

    delay_to_regression_bins_1d = {
        delay: create_estimator_bins_1d(delay)
        for delay in delay_to_Xy_1d.keys()
    }

    delay_to_score_bins_1d = {}

    for num_days, reg_bin in delay_to_regression_bins_1d.items():
        X, y = delay_to_Xy_1d[num_days]
        if CV:
            n_samples = X.shape[0]
            time_series_split = model_selection.TimeSeriesSplit(
                n_splits=3, gap=n_samples // 40
            )
            scores = get_cv_1d().cross_validate(
                reg_bin,
                X,
                y,
                cv=time_series_split,
            )
        else:
            scores = np.array([0])
        delay_to_score_bins_1d[num_days] = scores
        # label = get_label_1d(num_days, scores)
        # print('# ' + label.replace('$R^2_{oos}$ ', ''))
        _ = reg_bin.fit(X, y)

    dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
    dt_to_str = str(dt_to.date()) if dt_to is not None else '***'

    for (num_days, reg_bin), ax in zip(
        delay_to_regression_bins_1d.items(), axes
    ):
        scores = delay_to_score_bins_1d[num_days]
        style = delay_to_style[num_days]
        if num_color is not None:
            style += f'C{num_color}'
        plot_model_1d(
            reg_bin,
            ax,
            style,
            label=f'{dt_from_str} -- {dt_to_str}',
            relative=relative,
            **kwargs,
        )
        title = (
            f'{indicator_field} -> {profit_field}  '
            f'{get_label_1d(num_days, scores)}'
        )
        if relative is not None:
            title += f', relative to {relative}'
        ax.set_title(title)
        if y_ticks_interval is not None:
            assert isinstance(y_ticks_interval, typing.SupportsFloat)
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(y_ticks_interval)
            )
        # ax.legend()
    # plt.show()


DATE_RANGES = (
    (datetime(2015, 6, 1, 0, 0), datetime(2016, 6, 1, 0, 0)),
    (datetime(2016, 6, 1, 0, 0), datetime(2017, 6, 1, 0, 0)),
    (datetime(2017, 6, 1, 0, 0), datetime(2018, 6, 1, 0, 0)),
    (datetime(2018, 6, 1, 0, 0), datetime(2019, 6, 1, 0, 0)),
    (datetime(2019, 6, 1, 0, 0), datetime(2020, 6, 1, 0, 0)),
    (datetime(2020, 6, 1, 0, 0), datetime(2021, 6, 1, 0, 0)),
)


def plot_facet(
    *,
    indicator_field,
    profit_fields=('profit_in_currency', 'profit'),
    figsize=(28, 8),
    **kwargs,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(profit_fields),
        ncols=3,
        sharex=True,
        sharey=False,
        squeeze=True,
        figsize=figsize,
    )

    for i_profit_field, profit_field in enumerate(profit_fields):
        for i_date_range, (dt_from, dt_to) in enumerate(DATE_RANGES):
            plot_regressions_1d(
                dt_from=dt_from,
                dt_to=dt_to,
                indicator_field=indicator_field,
                profit_field=profit_field,
                num_color=i_date_range,
                axes=axes[i_profit_field],
                **kwargs,
            )

        plot_regressions_1d(
            dt_from=None,
            dt_to=None,
            indicator_field=indicator_field,
            profit_field=profit_field,
            axes=axes[i_profit_field],
            linewidth=3,
            color='white',
            **kwargs,
        )
    plt.show()


# indicator_1h
# indicator_1d
# indicator
# indicator_72d
# profit
# profit_in_currency


# %%
plot_facet(
    indicator_field='indicator_4h',
    relative=0,
    y_ticks_interval=0.1,
)

# %%
plot_facet(
    indicator_field='indicator_12h',
    y_ticks_interval=0.1,
    relative=0,
)

# %%
plot_facet(
    indicator_field='indicator_1d',
    y_ticks_interval=0.1,
    relative=0,
)

# %%
plot_facet(
    indicator_field='indicator_5d',
    min_x=-0.9,
    max_x=+0.9,
    y_ticks_interval=0.1,
    # relative=0,
    figsize=(28, 16),
)

# %%
plot_facet(
    indicator_field='indicator',
    min_x=-0.9,
    max_x=+0.9,
    y_ticks_interval=0.1,
    # relative=0,
    figsize=(28, 16),
)

# %%
plot_facet(
    indicator_field='indicator_72d',
    min_x=-0.75,
    max_x=+0.75,
    y_ticks_interval=0.1,
    relative=0,
    figsize=(28, 16),
)

# %%
plot_facet(
    indicator_field='market_indicator',
    min_x=-0.15,
    max_x=+0.10,
    y_ticks_interval=0.1,
    relative=0,
    figsize=(28, 16),
)


# %%
def separate_features_2d(
    delay,
    dt_from,
    dt_to,
    indicator_1_field,
    indicator_2_field,
    profit_field,
):
    return (
        _separate_indicators_2d(
            delay,
            dt_from,
            dt_to,
            indicator_1_field,
            indicator_2_field,
        ),
        _separate_profit_2d(
            delay,
            dt_from,
            dt_to,
            profit_field,
        )
    )


@control_output
@memory_.cache
def _separate_indicators_2d(
    delay,
    dt_from,
    dt_to,
    indicator_1_field,
    indicator_2_field,
):
    mask = _mask_1d(delay=delay, dt_from=dt_from, dt_to=dt_to)
    df = delay_to_df[delay]
    if mask is not None:
        df = df[mask]
    return df[[indicator_1_field, indicator_2_field]].values


@control_output
@memory_.cache
def _separate_profit_2d(
    delay,
    dt_from,
    dt_to,
    profit_field,
):
    mask = _mask_1d(delay=delay, dt_from=dt_from, dt_to=dt_to)
    df = delay_to_df[delay]
    if mask is not None:
        df = df[mask]
    return df[profit_field].values


# %% pycharm={"name": "#%%\n"}
# # %%timeit -n1 -r1

from datetime import datetime
import functools

from sklearn import ensemble
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

CV = False
CV_USES_HISTOGRAM = True
CACHE = True


def create_estimator_bins_2d(delay):
    radius = 0.12

    def _w(d):
        return norm.pdf(d / radius)

    return histogram.Histogram2dRegressionWrapper(
        bins=(40, 40, 1),
        shuffle=False,
        memory_=memory_,
        verbose=False,
        cache=CACHE,
        regressor=KNeighborsWeightedRegressor(
            n_neighbors=256,
            weights=_w,
            n_jobs=-1,
        ),
    )


def get_cv_2d():
    if CV_USES_HISTOGRAM:
        return cross_validation.CrossValidation(
            test_histogram_bins=(80, 80, 20),
            verbose=False,
            cache=CACHE,
            memory_=memory_,
        )
    else:
        return cross_validation.CrossValidation(
            verbose=False,
            cache=CACHE,
            memory_=memory_,
        )


def get_label_2d(num_days, scores):
    return (
        f'Ожидаемый доход, {num_days:<3} д. '
        # f'$R^2_{{oos}}$ {scores.mean(): .5f} ± {scores.std():.5f}'
    )


def plot_model_2d(
    ax, reg_k, *args, min_x0=-1, max_x0=+1, min_x1, max_x1, title
):
    x = np.linspace(min_x0, max_x0, num=100)
    y = np.linspace(min_x1, max_x1, num=100)
    g = np.meshgrid(x, y)
    X_pred = np.array(g).reshape(2, -1).T
    y_pred = reg_k.predict(X_pred)

    X, Y = g
    assert X.shape == Y.shape
    Z = y_pred.reshape(X.shape)

    v_min_color = -2.0
    v_max_color = 2.0
    v_step_color = 0.05
    v_num_color = 1 + int(round((v_max_color - v_min_color) / v_step_color))

    v_min_line = -2.0
    v_max_line = 2.0
    v_step_line = 0.20
    v_num_line = 1 + int(round((v_max_line - v_min_line) / v_step_line))

    color_norm = TwoSlopeNorm(0, v_min_color, v_max_color)
    CS = ax.contourf(
        X,
        Y,
        Z,
        levels=np.linspace(
            v_min_color,
            v_max_color,
            num=v_num_color,
        ),
        norm=color_norm,
        cmap='RdBu',
    )
    ax.set_title(title)
    CS2 = ax.contour(
        CS,
        levels=np.linspace(
            v_min_line,
            v_max_line,
            num=v_num_line,
        ),
        colors='black',
        linewidths=2,
    )
    ax.clabel(CS2, colors='black', fontsize=16)


def plot_regressions_2d(
    dt_from,
    dt_to,
    indicator_1_field,
    indicator_2_field,
    profit_field,
):
    delay_to_Xy_2d = {
        delay: separate_features_2d(
            delay=delay,
            dt_from=dt_from,
            dt_to=dt_to,
            indicator_1_field=indicator_1_field,
            indicator_2_field=indicator_2_field,
            profit_field=profit_field,
        )
        for delay, df in delay_to_df.items()
        # if delay == 180
    }
    delay_to_regression_bins_2d = {
        delay: create_estimator_bins_2d(delay)
        for delay in delay_to_Xy_2d.keys()
    }
    delay_to_score_bins_2d = {}

    for num_days, reg_bin in delay_to_regression_bins_2d.items():
        X, y = delay_to_Xy_2d[num_days]
        if CV:
            n_samples = X.shape[0]
            time_series_split = model_selection.TimeSeriesSplit(
                n_splits=3, gap=n_samples // 40
            )
            scores = get_cv_2d().cross_validate(
                reg_bin,
                X,
                y,
                cv=time_series_split,
            )
        else:
            scores = np.array([0])
        delay_to_score_bins_2d[num_days] = scores
        # label = get_label_2d(num_days, scores)
        # print('# ' + label.replace('$R^2_{oos}$ ', ''))
        _ = reg_bin.fit(X, y)

    num_subplots = len(delay_to_regression_bins_2d)
    fig, ax = plt.subplots(
        1, num_subplots, figsize=((9 + 1) * num_subplots, 9)
    )
    # fig.tight_layout()

    dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
    dt_to_str = str(dt_to.date()) if dt_to is not None else '***'
    fig.suptitle(
        f'{dt_from_str} -- {dt_to_str}   '
        f'{indicator_1_field} x {indicator_2_field} -> {profit_field}',
        fontsize=16,
    )
    for i, (num_days, reg_bin) in enumerate(
        delay_to_regression_bins_2d.items()
    ):
        scores = delay_to_score_bins_2d[num_days]
        style = delay_to_style[num_days]
        plot_model_2d(
            ax[i],
            reg_bin,
            style,
            min_x0=-0.85,
            max_x0=+0.85,
            min_x1=-1,
            max_x1=+1,
            title=get_label_2d(num_days, scores),
        )
    plt.show()


# indicator_1h
# indicator_1d

# 24d
# indicator

# indicator_72d
# profit
# profit_in_currency

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field='indicator',
        indicator_2_field='indicator_4h',
        profit_field='profit_in_currency',
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field='indicator',
    indicator_2_field='indicator_4h',
    profit_field='profit_in_currency',
)

# %%
min_v_01d = 0.25
min_v_72d = 0.05

month = 6

for delay in (7, 30, 180):
    print(delay)
    for year in (2015, 2016, 2017, 2018, 2019, 2020):
        df_all = delay_to_df[delay]
        print(f'    {year}.{month} -- {year + 1}.{month}')
        time_mask = df_all['t'].between(
            datetime(year, month, 1, 0, 0).timestamp(),
            datetime(year + 1, month, 1, 0, 0).timestamp() - 1,
        )
        df = df_all[time_mask]

        hot_mask = (df['indicator_4h'] < -min_v_01d) & (
            df['indicator_72d'] > min_v_72d
        )
        col_mask = (df['indicator_4h'] > +min_v_01d) & (
            df['indicator_72d'] > min_v_72d
        )

        print(
            f'    hot:  {df[hot_mask]["profit"].mean():.2f}, '
            f'freq: {hot_mask.sum() / len(df):.3f}'
        )
        print(
            f'    cold: {df[col_mask]["profit"].mean():.2f}, '
            f'freq: {col_mask.sum() / len(df):.3f}'
        )
        print(
            f'    h+c:  {df[col_mask|hot_mask]["profit"].mean():.2f}, '
            f'freq: {(col_mask.sum() + hot_mask.sum()) / len(df):.3f}'
        )
        print(f'    oth:  {df[~(col_mask|hot_mask)]["profit"].mean():.2f}')
        print()


# %%
# # %%timeit -n1 -r1
# 37.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

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

    for i, (num_days, (X, y)) in enumerate(delay_to_Xy_1d.items()):
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

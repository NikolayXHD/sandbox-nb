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
    df = pd.concat(
        pd.read_feather(file) for file in directory.glob('*.feather')
    )
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
def plot_2d_hist(df_k, n_days):
    x_field = 'indicator_72d'
    y_field = 'profit'
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

plot_2d_hist(delay_to_df[180], 180)


# %%
def separate_features_1d(df_i):
    return (
        df_i[['indicator_1d']].values,
        df_i['profit'].values,
    )

# indicator_1h
# indicator_1d
# indicator
# indicator_72d

delay_to_Xy_1d = {
    delay: separate_features_1d(df) for delay, df in delay_to_df.items()
    # if delay == 180
}

# %%
from sklearn import ensemble
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm

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
        regressor = KNeighborsWeightedRegressor(
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

def get_label(num_days, scores):
    return (
        f'Ожидаемый доход, {num_days:<3} д. '
        f'$R^2_{{oos}}$ {scores.mean(): .5f} ± {scores.std():.5f}'
    )


delay_to_regression_bins_1d = {
    delay: create_estimator_bins_1d(delay) for delay in delay_to_Xy_1d.keys()
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
    label = get_label(num_days, scores)
    print('# ' + label.replace('$R^2_{oos}$ ', ''))
    _ = reg_bin.fit(X, y)


# %%
def plot_model_1d(reg_k, *args, min_x=-1, max_x=+1, **kwargs):
    X_pred = np.linspace(min_x, max_x, num=1000)
    y_pred = reg_k.predict(X_pred.reshape(-1, 1))
    plt.plot(X_pred, y_pred, *args, **kwargs)
    plt.gca().grid(True)

for num_days, reg_bin in delay_to_regression_bins_1d.items():
    scores = delay_to_score_bins_1d[num_days]
    style = delay_to_style[num_days]
    plot_model_1d(
        reg_bin,
        style,
        min_x=-0.6,
        max_x=+0.6,
        label=get_label(num_days, scores),
    )

fig = plt.gca().figure
fig.legend()
fig.set_figwidth(18)
fig.set_figheight(6)
plt.show()

# %%
from datetime import datetime

def separate_features_2d(df_i):
    dt_from = datetime(2016, 1, 1, 0, 0).timestamp()
    dt_to = datetime(2018, 1, 1, 0, 0).timestamp()
    df = df_i[df_i['t'].between(dt_from, dt_to)]
    return (
        df[['indicator_72d', 'indicator_1d']].values,
        df['profit'].values,
    )

# indicator_1h
# indicator_1d
# indicator
# indicator_72d

delay_to_Xy_2d = {
    delay: separate_features_2d(df) for delay, df in delay_to_df.items()
    # if delay == 180
}

# %% pycharm={"name": "#%%\n"}
# # %%timeit -n1 -r1

from sklearn import ensemble
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm

CV = False
CV_USES_HISTOGRAM = True
CACHE = True


def create_estimator_bins_2d(delay):
    radius = 0.08

    def _w(d):
        return norm.pdf(d / radius)

    return histogram.Histogram2dRegressionWrapper(
        bins=(40, 40, 1),
        shuffle=False,
        memory_=memory_,
        verbose=False,
        cache=CACHE,
        regressor = KNeighborsWeightedRegressor(
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


def get_label(num_days, scores):
    return (
        f'Ожидаемый доход, {num_days:<3} д. '
        f'$R^2_{{oos}}$ {scores.mean(): .5f} ± {scores.std():.5f}'
    )


delay_to_regression_bins_2d = {
    delay: create_estimator_bins_2d(delay) for delay in delay_to_Xy_2d.keys()
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
    label = get_label(num_days, scores)
    print('# ' + label.replace('$R^2_{oos}$ ', ''))
    _ = reg_bin.fit(X, y)

# %%
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm


def plot_model_2d(reg_k, *args, min_x0=-1, max_x0=+1, min_x1, max_x1, title):
    x = np.linspace(min_x0, max_x0, num=100)
    y = np.linspace(min_x1, max_x1, num=100)
    g = np.meshgrid(x, y)
    X_pred = np.array(g).reshape(2, -1).T
    y_pred = reg_k.predict(X_pred)
    
    X, Y = g
    assert X.shape == Y.shape
    Z = y_pred.reshape(X.shape)    
    fig, ax = plt.subplots()
    
    v_min_color = -0.2
    v_max_color = 0.5
    v_step_color = 0.01
    v_num_color = 1 + int(round((v_max_color - v_min_color) / v_step_color))
    
    v_min_line = -0.2
    v_max_line = 0.5
    v_step_line = 0.05
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
    ax.figure.set_figwidth(14)
    ax.figure.set_figheight(14)
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
    plt.show()


for num_days, reg_bin in delay_to_regression_bins_2d.items():
    scores = delay_to_score_bins_2d[num_days]
    style = delay_to_style[num_days]
    plot_model_2d(
        reg_bin,
        style,
        min_x0=-0.2,
        max_x0=+0.2,
        min_x1=-0.6,
        max_x1=+0.6,
        title=get_label(num_days, scores),
    )

# %%

min_v_01d = 0.20
min_v_72d = 0.10

month = 6

for delay in (7, 30, 180):
    print(delay)
    for year in (2015, 2016, 2017, 2018, 2019, 2020):
        df_all = delay_to_df[delay]
        print(f'    {year}.{month} -- {year + 1}.{month}')
        time_mask = df_all['t'].between(
            datetime(year, month, 1, 0, 0).timestamp(),
            datetime(year + 1, month, 1, 0, 0).timestamp() - 1
        )
        df = df_all[time_mask]

        hot_mask = (df['indicator_1d'] < -min_v_01d) & (df['indicator_72d'] > min_v_72d)
        col_mask = (df['indicator_1d'] > +min_v_01d) & (df['indicator_72d'] > min_v_72d)

        print(f'    hot:  {df[hot_mask]["profit"].mean():.2f}, freq: {hot_mask.sum() / len(df):.3f}')
        print(f'    cold: {df[col_mask]["profit"].mean():.2f}, freq: {col_mask.sum() / len(df):.3f}')
        print(f'    h+c:  {df[col_mask|hot_mask]["profit"].mean():.2f}, freq: {(col_mask.sum() + hot_mask.sum()) / len(df):.3f}')
        print(f'    oth:  {df[~(col_mask|hot_mask)]["profit"].mean():.2f}')
        print()

# %%
# # %%timeit -n1 -r1
# 37.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

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

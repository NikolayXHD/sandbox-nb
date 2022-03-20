# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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
    df = df[df['list_level'] <= 2]
    df.drop('list_level', axis=1, inplace=True)
    df.rename({'annual_total_profit_ln': 'profit'}, axis=1, inplace=True)
    df.sort_values(by=['t'], inplace=True)
    return df


def separate_features(df_i):
    return (
        df_i['indicator'].values.reshape(-1, 1),
        df_i['profit'].values,
    )


def plot_2d_hist(df_k, n_days):
    ax = sns.histplot(data=df_k, x='indicator', y='profit', bins=(200, 100))
    ax.figure.set_figwidth(18)
    ax.figure.set_figheight(9)
    ax.grid(True)
    # ax.set_axisbelow(True)
    ax.set_title(f'доход через {n_days} дней, в пересчёте на 1 год')
    plt.show()


def plot_model(reg_k, *args, min_x=-1, max_x=+1, **kwargs):
    X_pred = np.linspace(min_x, max_x, num=1000)
    y_pred = reg_k.predict(X_pred.reshape(-1, 1))
    plt.plot(X_pred, y_pred, *args, **kwargs)
    plt.gca().grid(True)


delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

PWD = Path(os.path.dirname(os.path.realpath('__file__')))
CACHE_STORAGE_PATH = PWD.joinpath('..', '..', '.storage', 'cache')
OUTPUT_STORAGE_PATH = PWD.joinpath(
    '..', '..', '..', 'sandbox', '.storage', 'output'
)

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2019-07-01',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

delay_to_df = {delay: build_df(path) for delay, path in delay_to_dir.items()}

delay_to_Xy = {
    delay: separate_features(df) for delay, df in delay_to_df.items()
}

# %% pycharm={"name": "#%%\n"}
# # %%timeit -n1 -r1
# 37.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# Проверим, похожи ли огрубленные гистограммы, которые мы передадим
# в модель, на исходные.

fig, axes = plt.subplots(1, len(delay_to_Xy), figsize=(24, 7))
fig.suptitle('Дискретизированные значения')

bins = (50, 50)

for i, (num_days, (X, y)) in enumerate(delay_to_Xy.items()):
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

# %% pycharm={"name": "#%%\n"}
# # %%timeit -n1 -r1

# CV = False
# CV_USES_HISTOGRAM = True
# CACHE = False
# 27.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# CACHE = True
# 31.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# 2.76 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

from sklearn import ensemble

delay_to_kwargs = {
    7: dict(min_weight_fraction_leaf=0.0010),
    30: dict(min_weight_fraction_leaf=0.0010),
    180: dict(min_weight_fraction_leaf=0.0010),
}

CV = False
CV_USES_HISTOGRAM = True
CACHE = True


def create_estimator_bins(delay):
    return histogram.Histogram2dRegressionWrapper(
        bins=(200, 1),
        shuffle=False,
        memory_=memory_,
        verbose=False,
        cache=CACHE,
        regressor=ensemble.ExtraTreesRegressor(
            n_estimators=100,
            bootstrap=True,
            max_samples=0.5,
            n_jobs=-1,
            **delay_to_kwargs[delay],
        ),
    )


def get_cv():
    if CV_USES_HISTOGRAM:
        return cross_validation.CrossValidation(
            test_histogram_bins=(200, 200),
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


delay_to_regression_bins = {
    delay: create_estimator_bins(delay) for delay in (7, 30, 180)
}

delay_to_score_bins = {}

for num_days, reg_bin in delay_to_regression_bins.items():

    X, y = delay_to_Xy[num_days]
    if CV:
        n_samples = X.shape[0]
        time_series_split = model_selection.TimeSeriesSplit(
            n_splits=3, gap=n_samples // 40
        )
        scores = get_cv().cross_validate(
            reg_bin,
            X,
            y,
            cv=time_series_split,
        )
    else:
        scores = np.array([0])
    delay_to_score_bins[num_days] = scores
    label = get_label(num_days, scores)
    print('# ' + label.replace('$R^2_{oos}$ ', ''))
    _ = reg_bin.fit(X, y)

for num_days, reg_bin in delay_to_regression_bins.items():
    scores = delay_to_score_bins[num_days]
    style = delay_to_style[num_days]
    plot_model(
        reg_bin,
        style,
        #         min_x=-0.5,
        #         max_x=+0.5,
        label=get_label(num_days, scores),
    )

fig = plt.gca().figure
fig.legend()
fig.set_figwidth(18)
fig.set_figheight(6)
plt.show()

# extra_tree hist:no  cv_n_splits: 3
# Ожидаемый доход, 7   д.  0.00056 ± 0.00065
# Ожидаемый доход, 30  д.  0.00063 ± 0.00086
# Ожидаемый доход, 180 д.  0.00303 ± 0.00288

# extra_tree hist:yes  cv_n_splits: 3
# Ожидаемый доход, 7   д.  0.00061 ± 0.00035
# Ожидаемый доход, 30  д.  0.00038 ± 0.00069
# Ожидаемый доход, 180 д.  0.00263 ± 0.00174

# k_nearest  hist:yes  cv_n_splits: 3
# Ожидаемый доход, 7   д. -0.00022 ± 0.00072
# Ожидаемый доход, 30  д. -0.00056 ± 0.00133
# Ожидаемый доход, 180 д. -0.00088 ± 0.00328

# k_nearest  hist:no  cv_n_splits: 3
# Ожидаемый доход, 7   д. -0.00023 ± 0.00072
# Ожидаемый доход, 30  д. -0.00057 ± 0.00132
# Ожидаемый доход, 180 д. -0.00089 ± 0.00324

# k_nearest  hist:yes  cv_n_splits: 4
# Ожидаемый доход, 7   д. -0.00023 ± 0.00072
# Ожидаемый доход, 30  д. -0.00057 ± 0.00132
# Ожидаемый доход, 180 д. -0.00089 ± 0.00324

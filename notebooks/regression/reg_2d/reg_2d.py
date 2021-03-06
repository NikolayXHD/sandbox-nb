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

# %% pycharm={"name": "#%%\n"} tags=[]
from __future__ import annotations

from datetime import datetime
import typing

from regression.memory import control_output


def separate_features_2d(
    *,
    delay: int,
    date_from: datetime | None,
    date_to: datetime | None,
    indicator_1_field: str,
    indicator_2_field: str,
    profit_field: str,
    use_validation_df: bool,
):
    df = get_df(
        delay=delay,
        date_from=date_from,
        date_to=date_to,
        use_validation_df=use_validation_df,
    )
    return (
        df[[indicator_1_field, indicator_2_field]].values,
        df[profit_field].values,
        df['w'].values,
    )


# # %%timeit -n1 -r1

from datetime import datetime
import functools

from sklearn import ensemble
from regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

CACHE = True


def create_estimator_bins_2d(delay, radius, regression_bins=None):
    def _w(d):
        return norm.pdf(d / radius)

    if regression_bins is None:
        regression_bins = (40, 40, 1)

    return histogram.Histogram2dRegressionWrapper(
        bins=regression_bins,
        shuffle=False,
        memory_=memory_,
        verbose=False,
        cache=CACHE,
        regressor=KNeighborsWeightedRegressor(
            n_neighbors=128,
            weights=_w,
            n_jobs=-1,
        ),
    )


def plot_model_2d(
    ax,
    reg_k,
    *args,
    q1,
    title,
    hist_x,
    hist_w,
    min_x0: float,
    max_x0: float,
    min_x1: float,
    max_x1: float,
    alpha_min: float = 0.05,
    v_min_color: float = -5,
    v_max_color: float = +5,
    v_min_line: float = -10,
    v_max_line: float = +10,
    v_step_line: float = 0.2,
    bins: typing.Tuple[int, int] = (100, 100),
    levels=None,
    log_alpha_scale=True,
):
    if levels is None:
        v_num_line = int((v_max_line - v_min_line) // v_step_line)
        levels = np.linspace(v_min_line, v_max_line, num=v_num_line)

    hist, x_edges, y_edges = np.histogram2d(
        hist_x[:, 0],
        hist_x[:, 1],
        weights=hist_w,
        range=((min_x0, max_x0), (min_x1, max_x1)),
        bins=bins,
    )

    x = np.linspace(min_x0, max_x0, num=bins[0])
    y = np.linspace(min_x1, max_x1, num=bins[1])
    g = np.meshgrid(x, y)
    X, Y = g

    X_pred = np.array(g).reshape(2, -1).T
    X_pred_scaled = X_pred * np.array([[1, q1]])
    y_pred = reg_k.predict(X_pred_scaled)

    assert X.shape == Y.shape
    Z = y_pred.reshape(X.shape)

    hist_min = hist[hist > 0].min()
    alphas = (
        np.log(np.maximum(hist, hist_min) / hist_min)
        if log_alpha_scale
        else hist
    )
    alphas = np.minimum(
        1, alphas * ((1 - alpha_min) / alphas.max()) + alpha_min
    )
    alphas = alphas.T

    ax.imshow(
        np.zeros_like(Z),
        aspect=(max_x0 - min_x0) / (max_x1 - min_x1),
        extent=(min_x0, max_x0, min_x1, max_x1),
        cmap='gist_gray',
    )
    ax.imshow(
        Z,
        vmin=v_min_color,
        vmax=v_max_color,
        cmap='RdBu',
        aspect=(max_x0 - min_x0) / (max_x1 - min_x1),
        extent=(min_x0, max_x0, min_x1, max_x1),
        origin='lower',
        alpha=alphas,
    )
    ax.set_title(title)
    CS2 = ax.contour(
        Z,
        levels=levels,
        extent=(min_x0, max_x0, min_x1, max_x1),
        colors='black',
        linewidths=1.5,
    )
    ax.clabel(CS2, colors='black', fontsize=14)


def plot_regressions_2d(
    date_from: datetime,
    date_to: datetime,
    indicator_1_field: str,
    indicator_2_field: str,
    min_x0: float = -1,
    max_x0: float = +1,
    min_x1: float = -1,
    max_x1: float = +1,
    q1: float | None = None,
    radius: float | None = None,
    profit_field: str = 'profit_in_currency',
    ignore_ticker_weight: bool = False,
    use_validation_df: bool = False,
    regression_bins: typing.Tuple[int, int, int] | None = None,
    **kwargs,
) -> None:
    if radius is None:
        radius = 0.1
    if q1 is None:
        q1 = (max_x0 - min_x0) / (max_x1 - min_x1)

    delay_to_Xy_2d = {
        delay: separate_features_2d(
            delay=delay,
            date_from=date_from,
            date_to=date_to,
            indicator_1_field=indicator_1_field,
            indicator_2_field=indicator_2_field,
            profit_field=profit_field,
            use_validation_df=use_validation_df,
        )
        for delay, df in delay_to_df.items()
        # if delay == 180
    }
    delay_to_regression_bins_2d = {
        delay: create_estimator_bins_2d(
            delay,
            radius=radius,
            regression_bins=regression_bins,
        )
        for delay in delay_to_Xy_2d.keys()
    }

    for num_days, reg_bin in delay_to_regression_bins_2d.items():
        X, y, w = delay_to_Xy_2d[num_days]
        X_scaled = X * np.array([[1, q1]])
        if ignore_ticker_weight:
            w = None
        _ = reg_bin.fit(X_scaled, y, w)

    num_subplots = len(delay_to_regression_bins_2d)
    fig, ax = plt.subplots(
        1, num_subplots, figsize=((9 + 1) * num_subplots, 9)
    )
    fig.suptitle(
        f'{format_date(date_from)} -- {format_date(date_to)}   '
        f'{indicator_1_field} x {indicator_2_field} -> {profit_field}'
        f'{"   no ticker w" if ignore_ticker_weight else ""}',
        fontsize=16,
    )
    for i, (num_days, reg_bin) in enumerate(
        delay_to_regression_bins_2d.items()
    ):
        X, y, w = delay_to_Xy_2d[num_days]
        style = delay_to_style[num_days]
        plot_model_2d(
            ax[i],
            reg_bin,
            style,
            min_x0=min_x0,
            max_x0=max_x0,
            q1=q1,
            min_x1=min_x1,
            max_x1=max_x1,
            title=f'Expected income, {num_days:<3} d. ',
            hist_x=X,
            hist_w=w,
            **kwargs,
        )
    plt.show()


def plot_facet_2d(use_validation_df: bool = False, **kwargs):
    for date_from, date_to in iterate_date_ranges(
        append_empty_range=True, use_validation_df=use_validation_df
    ):
        plot_regressions_2d(
            date_from=date_from,
            date_to=date_to,
            use_validation_df=use_validation_df,
            **kwargs,
        )

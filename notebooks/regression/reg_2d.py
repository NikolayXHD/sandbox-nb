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
from regression.memory import control_output


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
        ),
        _separate_w_1d(delay=delay, dt_from=dt_from, dt_to=dt_to),
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
from regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

CACHE = True


def create_estimator_bins_2d(delay, radius):
    def _w(d):
        return norm.pdf(d / radius)

    return histogram.Histogram2dRegressionWrapper(
        bins=(40, 40, 1),
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
    min_x0=-1,
    max_x0=+1,
    q1,
    min_x1,
    max_x1,
    title,
):
    x = np.linspace(min_x0, max_x0, num=100)
    y = np.linspace(min_x1, max_x1, num=100)
    g = np.meshgrid(x, y)
    X_pred = np.array(g).reshape(2, -1).T
    X_pred_scaled = X_pred * np.array([[1, q1]])
    y_pred = reg_k.predict(X_pred_scaled)

    X, Y = g
    assert X.shape == Y.shape
    Z = y_pred.reshape(X.shape)

    v_min_color = -4.0
    v_max_color = +4.0
    v_step_color = 0.05
    v_num_color = 1 + int(round((v_max_color - v_min_color) / v_step_color))

    v_min_line = -10.0
    v_max_line = +10.0
    v_step_line = 0.40
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
    ignore_ticker_weight: bool = False,
    min_x0=-1,
    max_x0=+1,
    min_x1=-1,
    max_x1=+1,
    q1=1,
    radius=0.1,
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
        delay: create_estimator_bins_2d(delay, radius=radius)
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
    # fig.tight_layout()

    dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
    dt_to_str = str(dt_to.date()) if dt_to is not None else '***'
    fig.suptitle(
        f'{dt_from_str} -- {dt_to_str}   '
        f'{indicator_1_field} x {indicator_2_field} -> {profit_field}'
        f'{"   no ticker w" if ignore_ticker_weight else ""}',
        fontsize=16,
    )
    for i, (num_days, reg_bin) in enumerate(
        delay_to_regression_bins_2d.items()
    ):
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
        )
    plt.show()


# %%
indicator_1_fld = 'dln_exp_3d'
min_x0=-0.1
max_x0=+0.1


indicator_2_fld = 'dln_exp_no_vol_24d'
min_x1=-0.03
max_x1=+0.03
q1 = (max_x0 - min_x0) / (max_x1 - min_x1)

profit_fld = 'profit_in_currency'

radius=0.010

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
        radius=radius,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
    radius=radius,
)

# %%
indicator_1_fld = 'dln_exp_4h'
min_x0=-0.1
max_x0=+0.1


indicator_2_fld = 'dln_exp_no_vol_24d'
min_x1=-0.03
max_x1=+0.03
q1 = (max_x0 - min_x0) / (max_x1 - min_x1)

profit_fld = 'profit_in_currency'

radius=0.010

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
        radius=radius,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
    radius=radius,
)

# %%
indicator_1_fld = 'dln_exp_3d'
indicator_2_fld = 'dln_exp_no_vol_24d'
profit_fld = 'profit_in_currency'

min_x0=-0.1
max_x0=+0.3
min_x1=-0.05
max_x1=+0.075
q1 = 2
radius=0.02

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
        radius=radius,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
    radius=radius,
)

# %% pycharm={"name": "#%%\n"}
for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field='indicator_24d',
        indicator_2_field='ad_exp_24d',
        profit_field='profit_in_currency',
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field='indicator_24d',
    indicator_2_field='ad_exp_24d',
    profit_field='profit_in_currency',
)

# %%
min_x0=-1
max_x0=+1
min_x1 = -0.15
max_x1 = +0.25
q1 = 10

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field='indicator_24d',
        indicator_2_field='dln_exp_no_vol_24d',
        profit_field='profit_in_currency',
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field='indicator_24d',
    indicator_2_field='dln_exp_no_vol_24d',
    profit_field='profit_in_currency',
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
)

# %%
min_x0=-1
max_x0=+1
min_x1 = -0.15
max_x1 = +0.25
q1 = 10

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field='indicator_4h',
        indicator_2_field='dln_exp_no_vol_24d',
        profit_field='profit_in_currency',
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field='indicator_4h',
    indicator_2_field='dln_exp_no_vol_24d',
    profit_field='profit_in_currency',
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
)

# %%
indicator_1_fld = 'indicator_24d'
indicator_2_fld = 'dln_exp_24d'
profit_fld = 'profit_in_currency'

min_x0=-1
max_x0=+1
min_x1 = -0.15
max_x1 = +0.25
q1 = 10

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
)

# %%
indicator_1_fld = 'indicator_4h'
indicator_2_fld = 'dln_exp_24d'
profit_fld = 'profit_in_currency'

min_x0=-1
max_x0=+1
min_x1 = -0.15
max_x1 = +0.25
q1 = 10

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
)

# %%
indicator_1_fld = 'ad_exp_24d'
indicator_2_fld = 'dln_exp_24d'
profit_fld = 'profit_in_currency'

min_x0=-1
max_x0=+1
min_x1 = -0.15
max_x1 = +0.25
q1 = 10

for date_from, date_to in DATE_RANGES:
    plot_regressions_2d(
        dt_from=date_from,
        dt_to=date_to,
        indicator_1_field=indicator_1_fld,
        indicator_2_field=indicator_2_fld,
        profit_field=profit_fld,
        min_x0=min_x0,
        max_x0=max_x0,
        min_x1=min_x1,
        max_x1=max_x1,
        q1=q1,
    )

plot_regressions_2d(
    dt_from=None,
    dt_to=None,
    indicator_1_field=indicator_1_fld,
    indicator_2_field=indicator_2_fld,
    profit_field=profit_fld,
    min_x0=min_x0,
    max_x0=max_x0,
    min_x1=min_x1,
    max_x1=max_x1,
    q1=q1,
)

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
        _separate_w_1d(
            delay=delay, dt_from=dt_from, dt_to=dt_to
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


@control_output
@memory_.cache
def _separate_w_1d(*, delay, dt_from, dt_to):
    df = delay_to_df[delay]
    df_ticker = df[['ticker', 't']]
    mask = _mask_1d(delay=delay, dt_from=dt_from, dt_to=dt_to)
    if mask is not None:
        df_ticker = df_ticker[mask]
    
    df_agg = df_ticker.groupby('ticker').agg(
        **{
            't_min': pd.NamedAgg('t', 'min'),
            't_max': pd.NamedAgg('t', 'max'),
            'num_candles': pd.NamedAgg('t', 'count')
        }
    )
    df_agg['num_days'] = (df_agg['t_max'] - df_agg['t_min']) / (3600 * 24)
    df_agg['w'] = df_agg['num_days'] / df_agg['num_candles']
    df_merged = df_ticker[['ticker']].merge(
        df_agg[['w']],
        how='left',
        left_on='ticker',
        right_index=True,
        copy=False,
    )
    return df_merged['w'].values


# %%
from datetime import datetime
import typing

from matplotlib import ticker
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from sklearn import ensemble

CACHE = True


def create_estimator_bins_1d(delay, radius=None):
    if radius is None:
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


def plot_regressions_1d(
    *,
    dt_from,
    dt_to,
    indicator_field,
    profit_field,
    axes,
    num_color=None,
    relative: float | None = None,
    y_ticks_interval_minor: float | None = 0.05,
    y_ticks_interval_major: float | None = 0.2,
    ignore_ticker_weight: bool = False,
    radius=None,
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
        delay: create_estimator_bins_1d(delay, radius)
        for delay in delay_to_Xy_1d.keys()
    }

    for num_days, reg_bin in delay_to_regression_bins_1d.items():
        X, y, w = delay_to_Xy_1d[num_days]
        if ignore_ticker_weight:
            w = None
        reg_bin.fit(X, y, w)

    dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
    dt_to_str = str(dt_to.date()) if dt_to is not None else '***'

    for (num_days, reg_bin), ax in zip(
        delay_to_regression_bins_1d.items(), axes
    ):
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
        ax.grid(True, which='both')
        ax.grid(which='minor', alpha=0.25)
        title = (
            f'{indicator_field} -> {profit_field}  {num_days:<3} days'
        )
        if relative is not None:
            title += f', relative to {relative}'
        ax.set_title(title)

        if y_ticks_interval_minor is not None:
            assert isinstance(y_ticks_interval_minor, typing.SupportsFloat)
            ax.yaxis.set_minor_locator(
                ticker.MultipleLocator(y_ticks_interval_minor)
            )
        if y_ticks_interval_major is not None:
            assert isinstance(y_ticks_interval_major, typing.SupportsFloat)
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(y_ticks_interval_major)
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
    profit_fields=(
        'profit_in_currency', 
        # 'profit'
    ),
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
        current_axes = (
            axes[i_profit_field] if len(profit_fields) > 1 else axes
        )
        for i_date_range, (dt_from, dt_to) in enumerate(DATE_RANGES):
            plot_regressions_1d(
                dt_from=dt_from,
                dt_to=dt_to,
                indicator_field=indicator_field,
                profit_field=profit_field,
                num_color=i_date_range,
                axes=current_axes,
                **kwargs,
            )

        plot_regressions_1d(
            dt_from=None,
            dt_to=None,
            indicator_field=indicator_field,
            profit_field=profit_field,
            axes=current_axes,
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
)

# %%
plot_facet(
    indicator_field='ad_exp_4h',
    relative=0,
)

# %%
plot_facet(
    indicator_field='dln_exp_4h',
    relative=0,
    min_x=-0.15,
    max_x=+0.25,
    radius=0.02,
)

# %%
plot_facet(
    indicator_field='dln_exp_no_vol_4h',
    relative=0,
    min_x=-0.15,
    max_x=+0.25,
    radius=0.015,
)

# %%
plot_facet(
    indicator_field='indicator_3d',
    relative=0,
    figsize=(28, 10),
)

# %%
plot_facet(
    indicator_field='ad_exp_3d',
    relative=0,
    figsize=(28, 10),
)

# %%
plot_facet(
    indicator_field='dln_exp_3d',
    relative=0,
    figsize=(28, 10),
    min_x=-0.15,
    max_x=+0.25,
    radius=0.015,
)

# %%
plot_facet(
    indicator_field='dln_exp_no_vol_3d',
    relative=0,
    figsize=(28, 10),
    min_x=-0.15,
    max_x=+0.25,
    radius=0.01,
)

# %%
plot_facet(
    indicator_field='indicator_24d',
    relative=0,
    figsize=(28, 10),
)

# %%
plot_facet(
    indicator_field='ad_exp_24d',
    relative=0,
    figsize=(28, 10),
)

# %%
plot_facet(
    indicator_field='dln_exp_24d',
    relative=0,
    figsize=(28, 10),
    min_x=-0.15,
    max_x=+0.25,
    radius=0.01,
)

# %%
plot_facet(
    indicator_field='dln_exp_no_vol_24d',
    relative=0,
    figsize=(28, 10),
    min_x=-0.15,
    max_x=+0.25,
    radius=0.006,
)

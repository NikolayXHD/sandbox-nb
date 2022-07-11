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


def separate_features_1d(
    *,
    delay: int,
    date_from: datetime | None,
    date_to: datetime | None,
    indicator_field: str,
    profit_field: str,
    use_validation_df: bool = False,
):
    df = get_df(
        delay=delay,
        date_from=date_from,
        date_to=date_to,
        use_validation_df=use_validation_df,
    )
    return (
        df[[indicator_field]].values,
        df[profit_field].values,
        df['w'].values,
    )


# %%
from datetime import datetime
import typing

from matplotlib import ticker
from regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm
from sklearn import ensemble

CACHE = True


def create_estimator_bins_1d(
    delay: int,
    radius=None,
    regression_bins: typing.Tuple[int, int] | None = None,
):
    if radius is None:
        radius = 0.10
    if regression_bins is None:
        regression_bins = (40, 1)

    def _w(d):
        return norm.pdf(d / radius)

    return histogram.Histogram2dRegressionWrapper(
        bins=regression_bins,
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
    date_from,
    date_to,
    indicator_field,
    profit_field,
    axes,
    num_color=None,
    relative: float | None = None,
    y_ticks_interval_minor: float | None = 0.05,
    y_ticks_interval_major: float | None = 0.2,
    ignore_ticker_weight: bool = False,
    radius=None,
    use_validation_df: bool = False,
    regression_bins: typing.Tuple[int, int] | None = None,
    **kwargs,
):
    delay_to_regression_bins_1d = train_1d(
        date_from=date_from,
        date_to=date_to,
        indicator_field=indicator_field,
        profit_field=profit_field,
        ignore_ticker_weight=ignore_ticker_weight,
        radius=radius,
        regression_bins=regression_bins,
        use_validation_df=use_validation_df,
    )

    plot_1d(
        delay_to_regression_bins_1d,
        date_from=date_from,
        date_to=date_to,
        indicator_field=indicator_field,
        profit_field=profit_field,
        axes=axes,
        num_color=num_color,
        relative=relative,
        y_ticks_interval_minor=y_ticks_interval_minor,
        y_ticks_interval_major=y_ticks_interval_major,
        **kwargs,
    )


def train_1d(
    *,
    date_from,
    date_to,
    indicator_field,
    profit_field,
    ignore_ticker_weight: bool = False,
    radius=None,
    regression_bins: typing.Tuple[int, int] | None = None,
    use_validation_df: bool,
):
    delay_to_Xy_1d = {
        delay: separate_features_1d(
            delay=delay,
            date_from=date_from,
            date_to=date_to,
            indicator_field=indicator_field,
            profit_field=profit_field,
            use_validation_df=use_validation_df,
        )
        for delay, df in delay_to_df.items()
    }

    delay_to_regression_bins_1d = {
        delay: create_estimator_bins_1d(
            delay=delay,
            radius=radius,
            regression_bins=regression_bins,
        )
        for delay in delay_to_Xy_1d.keys()
    }

    for num_days, reg_bin in delay_to_regression_bins_1d.items():
        X, y, w = delay_to_Xy_1d[num_days]
        if ignore_ticker_weight:
            w = None
        reg_bin.fit(X, y, w)

    return delay_to_regression_bins_1d


def plot_1d(
    delay_to_regression_bins_1d,
    date_from,
    date_to,
    indicator_field,
    profit_field,
    axes,
    num_color,
    relative,
    y_ticks_interval_minor: float | None,
    y_ticks_interval_major: float | None,
    **kwargs,
):
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
            label=f'{format_date(date_from)} -- {format_date(date_to)}',
            relative=relative,
            **kwargs,
        )

        ax.grid(True, which='both')
        ax.grid(which='minor', alpha=0.25)
        title = f'{indicator_field} -> {profit_field}  {num_days:<3} days'
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


def plot_facet_1d(
    *,
    indicator_field,
    profit_fields=(
        'profit_in_currency',
        # 'profit'
    ),
    figsize=(28, 8),
    regression_bins: typing.Tuple[int, int | None] = None,
    use_validation_df: bool = False,
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
        current_axes = axes[i_profit_field] if len(profit_fields) > 1 else axes
        for i_date_range, (date_from, date_to) in enumerate(
            iterate_date_ranges(
                append_empty_range=True, use_validation_df=use_validation_df
            )
        ):
            if date_from is None:
                assert date_to is None
                kwargs_copy = {**kwargs, 'color': 'white', 'linewidth': 3}
            else:
                kwargs_copy = {**kwargs, 'num_color': i_date_range}

            plot_regressions_1d(
                date_from=date_from,
                date_to=date_to,
                indicator_field=indicator_field,
                profit_field=profit_field,
                axes=current_axes,
                regression_bins=regression_bins,
                use_validation_df=use_validation_df,
                **kwargs_copy,
            )

    plt.show()


# %%
plot_facet_1d(
    indicator_field='dln_exp_log_4h',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_log_4h',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_log_3d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_log_3d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_log_24d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_log_24d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_log_72d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_log_72d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='indicator_log_4h',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='ad_exp_log_4h',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='indicator_log_3d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='ad_exp_log_3d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='indicator_log_24d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='ad_exp_log_24d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='indicator_log_72d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='ad_exp_log_72d',
    min_x=-0.75,
    max_x=+0.75,
    radius=0.05,
    regression_bins=(100, 1),
)

# %% [markdown]
# Notice how in 7d profits, almost every year at the same values of
# `dln_exp_4h` at ±0.05, average profit sharply drops.
#
# Suggesting, "too oversold" / "too overbouht" penalties are nicely
# captured by indicator.

# %%
plot_facet_1d(
    indicator_field='dln_exp_4h',
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_4h',
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_3d',
    figsize=(28, 10),
    min_x=-0.09,
    max_x=+0.09,
    radius=0.003,
    regression_bins=(500, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_3d',
    figsize=(28, 10),
    min_x=-0.11,
    max_x=+0.11,
    radius=0.006,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_3d',
    figsize=(28, 10),
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_24d',
    figsize=(28, 10),
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %% tags=[]
plot_facet_1d(
    indicator_field='dln_exp_no_vol_24d',
    figsize=(28, 10),
    min_x=-0.03,
    max_x=+0.03,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_72d',
    figsize=(28, 10),
    min_x=-0.04,
    max_x=+0.04,
    radius=0.002,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_exp_no_vol_72d',
    figsize=(28, 10),
    min_x=-0.03,
    max_x=+0.03,
    radius=0.002,
    regression_bins=(250, 1),
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='indicator_4h',
    # relative=0,
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='ad_exp_4h',
    # relative=0,
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='indicator_3d',
    relative=-0.5,
    figsize=(28, 10),
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='ad_exp_3d',
    # relative=0,
    figsize=(28, 10),
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='indicator_24d',
    relative=+0.45,
    figsize=(28, 10),
)

# %% jupyter={"outputs_hidden": true} tags=[]
plot_facet_1d(
    indicator_field='ad_exp_24d',
    # relative=0,
    figsize=(28, 10),
)

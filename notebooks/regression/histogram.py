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
def plot_2d_hist(
    df_k,
    n_days,
    indicator_field,
    profit_field,
    figsize=(10, 10),
    ax=None
):
    x_field = indicator_field
    y_field = profit_field
    h, x_edges, y_edges = np.histogram2d(
        df_k[x_field], df_k[y_field], bins=(400, 400)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
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


# %%
fig, axes = plt.subplots(figsize=(20, 15), ncols=3)

for i, num_days in enumerate(delay_to_df.keys()):
    plot_2d_hist(
        delay_to_df[num_days],
        num_days,
        'dln_exp_24d',
        'profit_in_currency',
        ax=axes[i],
    )

plt.show()

# %%
fig, axes = plt.subplots(figsize=(20, 15), ncols=3)

for i, num_days in enumerate(delay_to_df.keys()):
    plot_2d_hist(
        delay_to_df[num_days],
        num_days,
        'dln_exp_no_vol_24d',
        'profit_in_currency',
        ax=axes[i],
    )

plt.show()

# %%
fig, axes = plt.subplots(figsize=(30, 8), ncols=3)

for i, num_days in enumerate(delay_to_df.keys()):
    plot_2d_hist(
        delay_to_df[num_days],
        num_days,
        'indicator_24d',
        'profit_in_currency',
        ax=axes[i],
    )

plt.show()

# %%
fig, axes = plt.subplots(figsize=(30, 8), ncols=3)

for i, num_days in enumerate(delay_to_df.keys()):
    plot_2d_hist(
        delay_to_df[num_days],
        num_days,
        'ad_exp_24d',
        'profit_in_currency',
        ax=axes[i],
    )

plt.show()

# %%
fig, ax = plt.subplots(figsize=(15, 15))

plot_2d_hist(
        delay_to_df[7],
        num_days,
        'indicator_24d',
        'ad_exp_24d',
        ax=ax,
    )

plt.show()

# %%
fig, ax = plt.subplots(figsize=(28, 8))

plot_2d_hist(
        delay_to_df[7],
        num_days,
        'indicator_4h',
        'dln_exp_no_vol_24d',
        ax=ax,
    )

plt.show()

# %%
fig, ax = plt.subplots(figsize=(25, 20))

plot_2d_hist(
        delay_to_df[7],
        num_days,
        'dln_exp_24d',
        'dln_exp_no_vol_24d',
        ax=ax,
    )

plt.show()


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

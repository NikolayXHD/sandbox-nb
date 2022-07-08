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
import math


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
):
    ax.grid(False)

    hist, xbins, ybins, im = ax.hist2d(
        df_k[indicator_field],
        df_k[profit_field],
        bins=bins,
        range=val_range,
        cmin=1,
        cmap='Blues',
    )
    
    if plot_values:
        x_delta = (xbins[-1] - xbins[0]) / len(xbins)
        y_delta = (ybins[-1] - ybins[0]) / len(ybins)
        for i in range(len(ybins) - 1):
            for j in range(len(xbins) - 1):
                if hist[j,i] > 0:
                    ax.text(
                        xbins[j] + x_delta / 2,
                        ybins[i] + y_delta / 2,
                        format_integer(round_integer(hist[j,i], 0)),
                        color='w',
                        fontsize=14,
                        bbox={'alpha': 0.25, 'facecolor': 'b'}
                        # fontweight="bold",
                    )
    
    if plot_xlabel:
        ax.set_xlabel(indicator_field)
    if plot_ylabel:
        ax.set_ylabel(profit_field + ' ' + str(n_days))
    ax.tick_params(axis='x',direction='in', pad=-12)
    ax.tick_params(axis='y',direction='in', pad=-22)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)


def round_integer(value, n):
    value_d = 10 ** math.floor(math.log10(value))
    return int(round(value / value_d, n) * value_d)


def format_integer(value) -> str:
    result = str(value) + '\0'
    result = result.replace('000000000000\0', 'T')
    result = result.replace('000000000\0', 'G')
    result = result.replace('000000\0', 'M')
    result = result.replace('000\0', 'K')
    result = result.replace('\0', '')
    return result


# %%
field_x = 'dln_exp_3d'
field_y = 'dln_exp_no_vol_24d'

fig, ax = plt.subplots(figsize=(15, 15))
for j, field in enumerate(fields):
    plot_2d_hist(
        delay_to_df[num_days],
        num_days,
        field_x,
        field_y,
        ax=ax,
        bins=(8, 6),
        val_range=((-0.1, +0.1), (-0.030, +0.030)),
        plot_values=True,
    )
plt.show()

# %%
fields = ('dln_exp_4h', 'dln_exp_no_vol_4h')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
        )

plt.show()

# %%
fields = ('dln_exp_3d', 'dln_exp_no_vol_3d')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
        )

plt.show()

# %%
fields = ('dln_exp_24d', 'dln_exp_no_vol_24d')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
        )

plt.show()

# %%
fields = ('indicator_4h', 'ad_exp_4h')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
        )

plt.show()

# %%
fields = ('indicator_3d', 'ad_exp_3d')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
        )

plt.show()

# %%
fields = ('indicator_24d', 'ad_exp_24d')

fig, axes = plt.subplots(figsize=(30, 15), nrows=3, ncols=2)
plt.subplots_adjust(wspace=0.01, hspace=0.03)

for i, num_days in enumerate(delay_to_df.keys()):
    for j, field in enumerate(fields):
        plot_2d_hist(
            delay_to_df[num_days],
            num_days,
            field,
            'profit_in_currency',
            ax=axes[i, j],
            plot_xlabel=i == len(delay_to_df) - 1,
            plot_ylabel=j == 0,
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

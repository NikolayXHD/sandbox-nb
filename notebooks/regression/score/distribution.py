# %%
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns


def get_high_score_distribution(score_min: float, score_max: float) -> pd.DataFrame:
    assert score_min <= score_max

    df = get_df(delay=7)
    s_t = df['t'].values.astype('datetime64[s]')
    df = df.assign(
        **{
            't_d': s_t.astype('datetime64[D]'), 't_y': s_t.astype('datetime64[Y]')
        }
    )
    df = df.loc[df['score'].between(score_min, score_max)]
    df_agg = df.groupby(['ticker', 't_d'], observed=True).agg(
        **{
            't_y': ('t_y', 'first'),
            'count': ('profit_in_currency', 'count'),
            'mean': ('profit_in_currency', 'mean'),
            'sum': ('profit_in_currency', 'sum'),
        }
    ).reset_index()

    return df_agg


def plot_high_score_distribution_df(df_agg: pd.DataFrame) -> None:
    day_min = df_agg['t_d'].min()
    day_max = df_agg['t_d'].max()
    df_agg_cut = df_agg.assign(
        **{
            'count_cat': pd.cut(df_agg['count'], 10),
            't_cat': pd.cut(df_agg['t_d'], 5)
        }
    )
    df_agg_cut.sort_values('t_cat', inplace=True)

    g = sns.FacetGrid(data=df_agg_cut, row='t_cat', height=4, aspect=20/4)
    g.map(sns.regplot, 'count', 'mean')

    for _, ax in g.axes_dict.items():
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.grid(which='minor', axis='y', visible=True, linewidth=0.5)

    plt.show()


def plot_high_score_distribution(score_min: float, score_max: float) -> None:
    df = get_high_score_distribution(score_min, score_max)
    plot_high_score_distribution_df(df)


# %% [markdown]
# ## Plot tables

# %%
df_agg = get_high_score_distribution(0.225, 1)
df_agg

# %%
for range_min, range_max in (
    (0, 250),
    (250, 500),
    (500, 1000),
):
    df_agg_range = df_agg.loc[df_agg['count'].between(range_min, range_max)]
    df_agg_range_days = df_agg_range.groupby(['t_y', 't_d'], observed=True).agg(
        **{
            'count': ('count', 'sum'),
            'sum': ('sum', 'sum'),
            'count_d_ticker': ('ticker', 'count'),
            'mean_d_ticker': ('mean', 'mean'),
        }
    )
    df_agg_range_days.loc[:, 'mean'] = (
        df_agg_range_days['sum'] / df_agg_range_days['count']
    )
    df_agg_range_days.reset_index(inplace=True)
    df_agg_range_years = df_agg_range_days.groupby('t_y', observed=True).agg(
        **{
            'count': ('count', 'sum'),
            'sum': ('sum', 'sum'),
            'count_d': ('t_d', 'count'),
            'adjusted_mean': ('mean_d_ticker', 'mean'),
        }
    )
    df_agg_range_years.loc[:, 'mean'] = (
        df_agg_range_years['sum'] / df_agg_range_years['count']
    )
    num_days = df_agg_range_years['count_d'].sum()
    adjusted_mean = df_agg_range_years['adjusted_mean'].mean()
    mean = (
        df_agg_range_years['sum'].sum() / df_agg_range_years['count'].sum()
    )
    print(
        f'{range_min} -- {range_max}: {num_days} days, '
        f'{mean=:.2f}  '
        f'{adjusted_mean=:.2f}'
    )
    display(df_agg_range_years.loc[:, ['count_d', 'mean', 'adjusted_mean']])
    # display(df_agg_range_days)
    print()

# %% [markdown]
# ## Plot regplots

# %%
plot_high_score_distribution(0.125, 0.175)

# %%
plot_high_score_distribution(0.175, 0.225)

# %%
plot_high_score_distribution(0.225, 1)

# %% [markdown]
# # leaders

# %%
for date_from, date_to in DATE_RANGES:
    print(f'{format_date(date_from)} -- {format_date(date_to)}')
    df = get_df(delay=7, date_from=date_from, date_to=date_to)
    with pd.option_context('display.min_rows', 50):
        s_counts = df.loc[df['score'] >= 0.225]['ticker'].value_counts()
        display(s_counts[s_counts > 0])
        print()

# %%
from __future__ import annotations

import pandas as pd
import numpy as np

df = get_df(delay=7)
s_t = df['t'].values.astype('datetime64[s]')
df = df.assign(
    **{
        't_d': s_t.astype('datetime64[D]'), 't_y': s_t.astype('datetime64[Y]')
    }
)
df = df.loc[df['score'] >= 0.225]
df_agg = df.groupby(['ticker', 't_d'], observed=True).agg(
    **{
        't_y': ('t_y', 'first'),
        'count': ('profit_in_currency', 'count'),
        'mean': ('profit_in_currency', 'mean'),
        'sum': ('profit_in_currency', 'sum'),
    }
).reset_index()

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

# %%
from matplotlib import pyplot as plt
import seaborn as sns

df

fig, ax = plt.subplots(figsize=(25, 6))
sns.regplot(data=df_agg, x='count', y='mean', ax=ax, order=3)
plt.show()

# %%
df_agg_cut = df_agg.assign(
    **{'count_cat': pd.cut(df_agg['count'], np.arange(0, 1000, 100))}
)

g = sns.catplot(
    data=df_agg_cut,
    x='count_cat',
    y='mean',
    row='t_y',
    kind='box',
    height=4,
    aspect=40/8,
)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(25, 6))
sns.violinplot(
    data=df_agg,
    x=pd.cut(df_agg['count'], np.arange(0, 1000, 50)),
    y='mean',
    scale='count',
    ax=ax,
)
plt.show()

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

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
df_agg_source = delay_to_df[180][['ticker', 't']]

df_agg = df_agg_source.groupby('ticker').agg(
    t_min=pd.NamedAgg('t', 'min'),
    t_max=pd.NamedAgg('t', 'max'),
    num_candles=pd.NamedAgg('t', 'count'),
)
df_agg['num_days'] = (df_agg['t_max'] - df_agg['t_min']) / (3600 * 24)
df_agg['w'] = df_agg['num_days'] / df_agg['num_candles']
df_agg_source.merge(df_agg[['w']], how='left', left_on='ticker', right_index=True, copy=False)


# df_agg.reset_index().join(df_agg_source)

# df_agg['num_days'] = (df_agg['t_max'] - df_agg['t_min']) / (3600 * 24)
# df_agg['num_days'].value_counts()

# %%
def plot_ticker_distribution(dt_from, dt_to):
    fig, ax = plt.subplots(figsize=(18, 6))
    df = delay_to_df[180]
    df = df[df['t'].between(dt_from.timestamp(), dt_to.timestamp())]
    val_counts = df['ticker'].value_counts()
    plt.bar(val_counts.index, val_counts.values)
    plt.xticks(rotation = 90)
    ax.set_title(f'{dt_from.date()} -- {dt_to.date()}')
    plt.show()

for dt_from, dt_to in DATE_RANGES:
    plot_ticker_distribution(dt_from, dt_to)

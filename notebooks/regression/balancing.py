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

# %% [markdown]
# $1$ наблюдение = $1$ свеча некоторого тикера $t$ в момент времени (интервал) $m$, обычно 1 минута
#
# Будем группировать интералы $m$ по целым дням $d$
#
# Без дополнительного взвешивания $\displaystyle w_{tm} = 1$
#
# $w_{d} = \displaystyle \sum_{t,m \in d} w_{tm} = \displaystyle \sum_{t,m} 1 = n_d$
#
# дни имеют разный суммарный вес
#
# Покажем, что суммарный вес всех дней станет однинаковый, если задать вес функцией вида $w_{tm}^* = \displaystyle \frac{f(n_{td})}{\displaystyle \sum_t n_{td} \cdot f(n_{td})}$
#
# $w_{d}^* = \displaystyle \sum_{t,m \in d} w^*_{tm} = \displaystyle \sum_{t,m \in d} \displaystyle \frac{f(n_{td})}{\displaystyle \sum_t n_{td} \cdot f(n_{td})} = \displaystyle \frac{\displaystyle \sum_{t,m \in d}f(n_{td})}{\displaystyle \sum_t n_{td} \cdot f(n_{td})} = \displaystyle \frac{\displaystyle \sum_t n_{td} \cdot f(n_{td})}{\displaystyle \sum_t n_{td} \cdot f(n_{td})} = 1$
#
# Выбор функции $f$ позволяет балансировать вес между тикерами в рамках 1 дня.
#
# В простейшем случае $f = 1 \implies \displaystyle w_{tm}^* = \displaystyle \frac{1}{\displaystyle \sum_t n_{td}} = \displaystyle \frac{1}{n_d}$
#
# Тогда суммарный вес тикера за 1 день $w_{td}^* = n_{td}$, то есть прямо пропорционален количеству свеч этого тикера в этот день.
#
# Подберём $f$, чтобы сгладить это различие до, например, $w_{td}^{**} = C \cdot \ln n_{td}$
#
# Этому условию удовлетворяет $f(n_{td}) = \displaystyle \frac{\ln n_{td}}{n_{td}}$, отсюда
#
# $ w_{tm}^{**} = \displaystyle \frac{\displaystyle \frac{\ln n_{td}}{n_{td}}}{\displaystyle \sum_{t} n_{td} \cdot \displaystyle \frac{\ln n_{td}}{n_{td}}} = \displaystyle \frac{\ln n_{td}}{n_{td} \cdot \displaystyle \sum_{t} \ln n_{td}}$
#
# Глядя на формулу $ w_{tm}^{**} $ можно понять, для чего приведено подробное обоснование. Ясно ли сходу, зачем в знаменателе первый множитель $n_{td}$?

# %%
df_agg_source = delay_to_df[180][['ticker', 't']]
df_agg_source = df_agg_source.assign(
    **{'d': df_agg_source['t'] // (3600 * 24)}
)

df_agg_day_ticker = (
    df_agg_source.groupby(['d', 'ticker'])
    .agg(**{'n_td': ('t', 'count')})
    .reset_index()
)

df_agg_day_ticker = df_agg_day_ticker.assign(
    **{'ln_n_td': np.log(1 + df_agg_day_ticker['n_td'])}
)

# %%
df_agg_day_ticker

# %%
df_agg_day = (
    df_agg_day_ticker.groupby('d')
    .agg(**{'sum_t_ln_n_td': ('ln_n_td', 'sum')})
    .reset_index()
)

df_agg_day

# %%
df_agg_result = df_agg_source.merge(df_agg_day, on='d', copy=False).merge(
    df_agg_day_ticker, on=('d', 'ticker'), copy=False
)

df_agg_result

# %%
df_agg_result = df_agg_result.assign(
    **{
        'w': df_agg_result['ln_n_td']
        / df_agg_result['n_td']
        / df_agg_result['sum_t_ln_n_td']
    }
)

df_agg_result

# %%
df_agg_result.groupby('d').agg(**{'w_d': ('w', 'sum')}).describe()

# %%
del df_agg_result, df_agg_day, df_agg_day_ticker, df_agg_source

# %%
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


def plot_days():
    days = df_agg_source['d'].value_counts()
    sns.lineplot(x=days.index, y=days.values)


plot_days()

# %%
df_agg[(df_agg['t_day'] == 16616) & df_agg['n'] > 0]

# %%
df_agg[(df_agg['t_day'] == 16618) & df_agg['n'] > 0]


# %%
def plot_ticker_distribution(date_from, date_to):
    fig, ax = plt.subplots(figsize=(30, 6))
    df = get_df(delay=180, date_from=date_from, date_to=date_to)
    val_counts = df['ticker'].value_counts()
    plt.bar(val_counts.index, val_counts.values)
    plt.xticks(rotation=90)
    ax.set_title(f'{format_date(date_from)} -- {format_date(date_to)}')
    plt.show()


for date_from, date_to in iterate_date_ranges():
    plot_ticker_distribution(date_from, date_to)

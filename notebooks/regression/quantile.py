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

# %%
from __future__ import annotations

import pandas as pd
import itertools


def build_df_indicator_quantiles(indicator_names: list[str]) -> pd.DataFrame:
    values = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    values_all = list(
        itertools.chain(values, (1 - v for v in reversed(values)))
    )
    indicator_name_to_q_values = {
        indicator_name: delay_to_df[180][indicator_name].quantile(values_all)
        for indicator_name in indicator_names
    }
    df = pd.DataFrame(
        {
            f'q_{value}': pd.Series(
                indicator_name_to_q_values[indicator_name].iloc[value_index]
                for indicator_name in indicator_names
            )
            for value_index, value in enumerate(values_all)
        }
    )
    df.index = pd.Index(indicator_names)
    return df


df_indicator_quantiles = build_df_indicator_quantiles(
    [
        'indicator_4h',
        'indicator_3d',
        'indicator_24d',
        'indicator_72d',
        'ad_exp_4h',
        'ad_exp_3d',
        'ad_exp_24d',
        'ad_exp_72d',
        'dln_exp_4h',
        'dln_exp_3d',
        'dln_exp_24d',
        'dln_exp_72d',
        'dln_exp_no_vol_4h',
        'dln_exp_no_vol_3d',
        'dln_exp_no_vol_24d',
        'dln_exp_no_vol_72d',
    ]
)

df_indicator_quantiles

# %%
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(data=df_indicator_quantiles, ax=ax)
plt.xticks(rotation=45)
plt.show()

# %%
df = delay_to_df[180]
df.describe()

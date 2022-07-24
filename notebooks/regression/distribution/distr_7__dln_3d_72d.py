# %% [markdown]
# ## Print tables

# %%
import typing


def print_value_duration_range_tables(
    ranges: list[typing.Tuple[float, float]]
) -> None:
    df_agg = get_high_score_distribution(0.225, 1, 'score-dln-3d-72d', delay=7)
    display(df_agg)
    for range_min, range_max in ranges:
        print_range_table(df_agg, range_min, range_max)


print_value_duration_range_tables(((0, 250), (250, 500), (500, 1000)))

# %% [markdown]
# # High scores

# %% [markdown]
# ## Plot regplots

# %%
plot_high_score_distribution(0.2, 0.25, 'score-dln-3d-72d', delay=7)

# %%
plot_high_score_distribution(0.25, 0.35, 'score-dln-3d-72d', delay=7)

# %%
plot_high_score_distribution(0.35, 1, 'score-dln-3d-72d', delay=7)

# %% [markdown]
# ## Print tables

# %%
import typing


def print_value_duration_range_tables(
    ranges: list[typing.Tuple[float, float]]
) -> None:
    df_agg = get_high_score_distribution(0.225, 1, 'score-dln-3-72', delay=7)
    display(df_agg)
    for range_min, range_max in ranges:
        print_range_table(df_agg, range_min, range_max)


print_value_duration_range_tables(((0, 250), (250, 500), (500, 1000)))

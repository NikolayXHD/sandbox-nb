# %% [markdown]
# # `dln_log_3d` x `dln_log_72d`
#
# higher frequency of high 30d profit
#
# ```
# frequency       | 180d  30d   7d    |
# ----------------+-------------------+
# 0.13            | +0.32 +0.47 +0.30 |
# 0.045           | +0.46 +0.72 +0.44 |
# 0.02            | +0.53 +0.92 +0.49 |
# ----------------+-------------------+
# ```

# %%
def sim_range(min_val, max_val):  # type: ignore[no-redef]
    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_log_3d']
        y = df['dln_log_72d']
        slope_bin = 0.25 / 8
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) & x.between(-1, 0.50)
        return score

    sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')


for min_val, max_val in (
    (-0.30, +0.35),
    (-0.40, +0.40),
    (-0.45, +0.45),
):
    sim_range(min_val, max_val)
    print()

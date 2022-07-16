# %% [markdown]
# # `dln_log_3d` x `dln_log_24d` H-like score

# %% [markdown]
# ## Validate on 2022 history
#
# it confidently repeats

# %%
for min_val, max_val in (
    (-0.30, 0.15),
    (-0.30, 0.25),
    (-0.40, 0.30),
    (-0.45, 0.45),
):

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_log_3d']
        y = df['dln_log_24d']
        slope_bin = 0.5 / 8  #
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) * x.between(-0.50, 0.50).astype('int')
        return score

    sim(
        _get_score,
        title=f'{min_val:.2f} -- {max_val:.2f}',
        use_validation_df=True,
    )
    print()

# %% [markdown]
# ## Narrowing ranges for `dln_log_24d`
#
# ```
# frequency       | 180d  30d   7d    |
# ----------------+-------------------+
# 0.16            | +0.37 +0.30 +0.22 |
# 0.05            | +0.75 +0.53 +0.40 |
# 0.016           | +1.51 +0.88 +0.56 |
# 0.006           | +2.21 +1.31 +0.57 |
# ----------------+-------------------+
# ```

# %%
for min_val, max_val in (
    (-0.30, 0.15),
    (-0.30, 0.25),
    (-0.35, 0.25),
    (-0.40, 0.30),
    (-0.45, 0.45),
):

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        x = df['dln_log_3d']
        y = df['dln_log_24d']
        slope_bin = 0.5 / 8  #
        bin_x = 0.2
        bin_y = 0.2
        min_x = -0.8
        score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
            min_val, max_val
        ) * x.between(-0.50, 0.50).astype('int')
        return score

    sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')
    print()


# %% [markdown]
# ## Slice `dln_log_3d`

# %%
def sim_ranges(val_min, val_max, step):  # type: ignore[no-redef]
    for val in np.arange(val_min, val_max, step):

        # type: ignore[no-redef]
        def _get_score(df: pd.DataFrame) -> np.ndarray:
            x = df['dln_log_3d']
            y = df['dln_log_24d']
            slope_bin = 0.5 / 8
            bin_x = 0.2
            bin_y = 0.2
            min_x = -0.8
            score = (
                (y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
                    val, val + step
                )
                # ~y.between(-0.30, 0.30)
                & x.between(-0.50, 0.50)
            )
            return score

        sim(_get_score, title=f'{val:.2f} -- {val + step:.2f}')
        print()


sim_ranges(-0.8, 0.8, 0.1)


# %% [markdown]
# ## Maximal 180d profit
#
# since extreme values of long indicator have high 180d income and low /
# negative 7d, 30d incomes, maximal 180d income is achieved at 4 corners

# %%
def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
    x = df['dln_log_3d']
    y = df['dln_log_24d']
    slope_bin = 0.5 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
        -0.30, 0.15
    ) & ~x.between(-0.25, 0.50).astype('int')
    return score


sim(_get_score, title=f'{-0.3:.2f} -- {0.2:.2f}')


# %% [markdown]
# ## Slice `dln_log_24d`

# %%
def sim_ranges(val_min, val_max, step):  # type: ignore[no-redef]
    for val in np.arange(val_min, val_max, step):

        # type: ignore[no-redef]
        def _get_score(df: pd.DataFrame) -> np.ndarray:
            x = df['dln_log_3d']
            y = df['dln_log_24d']
            slope_bin = 0.5 / 8
            bin_x = 0.2
            bin_y = 0.2
            min_x = -0.8
            score = ~(y - (bin_y / bin_x * slope_bin) * (x - min_x)).between(
                -0.3, 0.2
            ) & x.between(val, val + step)
            return score

        sim(_get_score, title=f'{val:.2f} -- {val + step:.2f}')
        print()


sim_ranges(-0.9, 0.9, 0.1)

# %% [markdown]
# # `dln_log_3d` x `dln_log_24d` hyperbolic score

# %% [markdown]
# ## chosen range

# %%
def sim_range(min_val: float, max_val: float):  # type: ignore[no-redef]

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        i1 = df['dln_log_3d'] / 0.7
        i2 = df['dln_log_24d'] / 0.6
        y = i2 - i1
        x = i2 + i1
        return (x * y).between(min_val, max_val)

    sim(_get_score, title=f'{min_val:.3f} -- {max_val:.3f}')


for min_val, max_val in (
    (-1, 0.075),
    (0.075, 0.100),
    (0.100, 0.125),
    (0.125, 0.225),
    (0.225, 1),
):
    sim_range(min_val, max_val)


# %% [markdown]
# ## ranges

# %%
def sim_ranges():  # type: ignore[no-redef]
    values = [-1, *np.arange(-0.25, 0.25, 0.025), 1]
    for min_val, max_val in zip(values, values[1:]):

        def _get_score(  # type: ignore[no-redef]
            df: pd.DataFrame,
        ) -> np.ndarray:
            i1 = df['dln_log_3d'] / 0.7
            i2 = df['dln_log_24d'] / 0.6
            y = i2 - i1
            x = i2 + i1
            return (x * y).between(min_val, max_val)

        sim(_get_score, title=f'{min_val:.3f} -- {max_val:.3f}')
        print()


sim_ranges()


# %% [markdown]
# ## chosen range

# %%
def sim_range():  # type: ignore[no-redef]
    min_val = 0.0215
    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        i1 = df['dln_log_3d'] * (0.54 / 0.7)
        i2 = df['dln_log_24d'] * (0.46 / 0.6)
        return (i2 ** 2 - i1 ** 2) > min_val

    sim(_get_score, title=f'{min_val:.2f} -- ***')


sim_range()


# %% [markdown]
# ## ranges

# %%
def sim_ranges():  # type: ignore[no-redef]
    values = [-1, *np.arange(-0.25, 0.25, 0.025), 1]
    for min_val, max_val in zip(values, values[1:]):

        def _get_score(  # type: ignore[no-redef]
            df: pd.DataFrame,
        ) -> np.ndarray:
            i1 = df['dln_log_3d'] / 0.7
            i2 = df['dln_log_24d'] / 0.6
            y = 0.46 * i2 - 0.54 * i1
            x = 0.46 * i2 + 0.54 * i1
            return (x * y).between(min_val, max_val)

        sim(_get_score, title=f'{min_val:.3f} -- {max_val:.3f}')
        print()


sim_ranges()

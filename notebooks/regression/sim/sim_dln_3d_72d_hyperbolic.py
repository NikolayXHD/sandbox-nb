# %% [markdown]
# # `dln_log_3d` x `dln_log_72d` hyperbolic score

# %% [markdown]
# ## Chosen range

# %%
def sim_range():  # type: ignore[no-redef]
    min_val = 0.20

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        i1 = df['dln_log_3d'] / 0.7
        i2 = df['dln_log_72d'] / 0.5
        return (i2 ** 2 - i1 ** 2) > min_val

    sim(_get_score, title=f'{min_val:.2f} -- ***')


sim_range()


# %% [markdown]
# ## Validate on data from 2022

# %%
def sim_range():  # type: ignore[no-redef]
    min_val = 0.20

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        i1 = df['dln_log_3d'] / 0.7
        i2 = df['dln_log_72d'] / 0.5
        return (i2 ** 2 - i1 ** 2) > min_val

    sim(_get_score, title=f'{min_val:.2f} -- ***', use_validation_df=True)


sim_range()


# %% [markdown]
# ## ranges

# %%
def sim_ranges():  # type: ignore[no-redef]
    values = np.arange(-1, 1, 0.1)
    for min_val, max_val in zip(values, values[1:]):

        def _get_score(  # type: ignore[no-redef]
            df: pd.DataFrame,
        ) -> np.ndarray:
            i1 = df['dln_log_3d'] / 0.7
            i2 = df['dln_log_72d'] / 0.5
            return (i2 ** 2 - i1 ** 2).between(min_val, max_val)

        sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')
        print()


sim_ranges()


# %% [markdown]
# # `dln_log_3d` x `dln_log_72d` hyperbolic score adjusted

# %% [markdown]
# ## chosen range

# %%
def sim_range():  # type: ignore[no-redef]
    min_val = 0.2

    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        i1 = df['dln_log_3d'] / 0.7
        i2 = df['dln_log_72d'] / 0.5
        x = i2 - i1
        y = 0.78 * i2 + 0.22 * i1
        return (x * y) > min_val

    sim(_get_score, title=f'{min_val:.2f} -- ***')


sim_range()


# %% [markdown]
# ## ranges

# %%
def sim_ranges():  # type: ignore[no-redef]
    values = np.arange(-1, 1, 0.1)
    for min_val, max_val in zip(values, values[1:]):

        def _get_score(  # type: ignore[no-redef]
            df: pd.DataFrame,
        ) -> np.ndarray:
            i1 = df['dln_log_3d'] / 0.7
            i2 = df['dln_log_72d'] / 0.5
            x = i2 - i1
            y = 0.78 * i2 + 0.22 * i1
            return (x * y).between(min_val, max_val)

        sim(_get_score, title=f'{min_val:.2f} -- {max_val:.2f}')
        print()


sim_ranges()

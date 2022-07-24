# %% [markdown]
# ## chosen range

# %%
def sim_range(min_val: float, max_val: float):  # type: ignore[no-redef]
    def _get_score(df: pd.DataFrame) -> np.ndarray:  # type: ignore[no-redef]
        score = df['dln-3d-dln-24d-0-exp-7d']
        return score.between(min_val, max_val)

    sim(_get_score, title=f'{min_val:.3f} -- {max_val:.3f}')


for min_val, max_val in (
    (-1, 0.05),
    (0.05, 0.075),
    (0.075, 0.10),
    (0.10, 0.15),
    (0.15, 1),
):
    sim_range(min_val, max_val)


# %% [markdown]
# ## ranges

# %%
def sim_ranges():  # type: ignore[no-redef]
    values = [-1, *np.arange(-0.25, 0.25, 0.05), 1]
    for min_val, max_val in zip(values, values[1:]):

        def _get_score(  # type: ignore[no-redef]
            df: pd.DataFrame,
        ) -> np.ndarray:
            score = df['dln-3d-dln-24d-0-exp-7d']
            return score.between(min_val, max_val)

        sim(_get_score, title=f'{min_val:.3f} -- {max_val:.3f}')
        print()


sim_ranges()

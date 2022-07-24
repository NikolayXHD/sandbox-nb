# %% [markdown]
# ## baseline score, no second exp smoothing

# %%
plot_facet_1d(
    indicator_field='score-dln-3d-24d',
    min_x=-0.4,
    max_x=+0.4,
    radius=0.05,
    regression_bins=(100, 1),
)

# %% [markdown]
# ## scores with additional exp smoothing

# %%
plot_facet_1d(
    indicator_field='dln-3d-dln-24d-0-exp-3d',
    min_x=-0.4,
    max_x=+0.4,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln-3d-dln-24d-0-exp-7d',
    min_x=-0.4,
    max_x=+0.4,
    radius=0.05,
    regression_bins=(100, 1),
)

# %%
plot_facet_1d(
    indicator_field='dlnv_4h',
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_4h',
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dlnv_3d',
    figsize=(28, 10),
    min_x=-0.09,
    max_x=+0.09,
    radius=0.003,
    regression_bins=(500, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_3d',
    figsize=(28, 10),
    min_x=-0.11,
    max_x=+0.11,
    radius=0.006,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dlnv_24d',
    figsize=(28, 10),
    min_x=-0.11,
    max_x=+0.11,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_24d',
    figsize=(28, 10),
    min_x=-0.03,
    max_x=+0.03,
    radius=0.004,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dlnv_72d',
    figsize=(28, 10),
    min_x=-0.04,
    max_x=+0.04,
    radius=0.002,
    regression_bins=(120, 1),
)

# %%
plot_facet_1d(
    indicator_field='dln_72d',
    figsize=(28, 10),
    min_x=-0.03,
    max_x=+0.03,
    radius=0.002,
    regression_bins=(250, 1),
)

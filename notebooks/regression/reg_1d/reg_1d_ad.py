# %%
plot_facet_1d(
    indicator_field='adv_4h',
    # relative=0,
)

# %%
plot_facet_1d(
    indicator_field='ad_4h',
    # relative=0,
)

# %%
plot_facet_1d(
    indicator_field='adv_3d',
    relative=-0.5,
    figsize=(28, 10),
)

# %%
plot_facet_1d(
    indicator_field='ad_3d',
    # relative=0,
    figsize=(28, 10),
)

# %%
plot_facet_1d(
    indicator_field='adv_24d',
    relative=+0.45,
    figsize=(28, 10),
)

# %%
plot_facet_1d(
    indicator_field='ad_24d',
    # relative=0,
    figsize=(28, 10),
)

# %%
plot_facet_1d(
    indicator_field='adv_72d',
    figsize=(28, 10),
    min_x=-0.4,
    max_x=+0.5,
    radius=0.01,
    regression_bins=(200, 1),
)

# %%
plot_facet_1d(
    indicator_field='ad_72d',
    figsize=(28, 10),
    min_x=-0.4,
    max_x=+0.5,
    radius=0.01,
    regression_bins=(200, 1),
)

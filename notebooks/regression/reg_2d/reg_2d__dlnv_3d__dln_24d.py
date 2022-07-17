# %%
plot_facet_2d(
    indicator_1_field='dlnv_3d',
    min_x0=-0.04,
    max_x0=+0.04,
    indicator_2_field='dln_24d',
    min_x1=-0.02,
    max_x1=+0.02,
    alpha_min=0.1,
    v_min_color=-1.5,
    v_max_color=+1.5,
    levels=(
        *np.linspace(-6.0, -2.8, num=5),
        *np.linspace(-2.0, -0.4, num=5),
        *(-0.2, -0.1, 0, +0.1, +0.2),
        *np.linspace(+0.4, +2.0, num=5),
        *np.linspace(+2.8, +6.0, num=5),
    ),
    regression_bins=(200, 200, 1),
    radius=0.001,
    log_alpha_scale=True,
)

# %%
plot_facet_2d(
    indicator_1_field='dlnv_3d',
    min_x0=-0.025,
    max_x0=+0.025,
    indicator_2_field='dln_24d',
    min_x1=-0.01,
    max_x1=+0.01,
    alpha_min=0.1,
    v_min_color=-1.5,
    v_max_color=+1.5,
    levels=(
        *np.linspace(-6.0, -2.8, num=5),
        *np.linspace(-2.0, -0.4, num=5),
        *(-0.2, -0.1, 0, +0.1, +0.2),
        *np.linspace(+0.4, +2.0, num=5),
        *np.linspace(+2.8, +6.0, num=5),
    ),
    regression_bins=(120, 120, 1),
    radius=0.002,
)

# %%
plot_facet_2d(
    indicator_1_field='dlnv_3d',
    min_x0=-0.1,
    max_x0=+0.1,
    indicator_2_field='dln_24d',
    min_x1=-0.04,
    max_x1=+0.04,
    radius=0.005,
    alpha_min=0.1,
    v_step_line=0.25,
    levels=(
        *np.linspace(-6.0, -2.8, num=5),
        *np.linspace(-2.0, -0.4, num=5),
        *(-0.2, -0.1, 0, +0.1, +0.2),
        *np.linspace(+0.4, +2.0, num=5),
        *np.linspace(+2.8, +6.0, num=5),
    ),
)

# %%
plot_facet_2d(
    indicator_1_field='dlnv_3d',
    min_x0=-0.1,
    max_x0=+0.1,
    indicator_2_field='dln_24d',
    min_x1=-0.04,
    max_x1=+0.04,
    radius=0.012,
    use_validation_df=True,
)

# %%
plot_facet_2d(
    indicator_1_field='dlnv_3d',
    min_x0=-0.1,
    max_x0=+0.1,
    indicator_2_field='dln_24d',
    min_x1=-0.04,
    max_x1=+0.04,
    radius=0.012,
)

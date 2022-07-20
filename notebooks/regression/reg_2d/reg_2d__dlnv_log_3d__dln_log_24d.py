# %%
plot_facet_2d(
    indicator_1_field='dlnv_log_3d',
    min_x0=-0.8,
    max_x0=+0.8,
    indicator_2_field='dln_log_24d',
    min_x1=-0.8,
    max_x1=+0.8,
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
    radius=0.02,
    log_alpha_scale=True,
)

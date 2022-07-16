# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# %% [markdown] tags=[]
# # Hyperbolic score

# %% tags=[]
def plot_score():
    i2, i1 = np.mgrid[-0.7:0.7:50j, -0.7:0.7:50j]

    j1 = i1 / 0.7
    j2 = i2 / 0.7

    x = 0.45 * j2 - 0.55 * j1
    y = 0.45 * j2 + 0.55 * j1

    score = x * y

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    ax.pcolormesh(
        i1,
        i2,
        score,
        cmap='RdBu',
        vmin=-0.3,
        vmax=+0.3,
    )
    ax.grid(True)

    cs = ax.contour(
        score,
        extent=(-0.7, +0.7, -0.7, +0.7),
        levels=np.arange(-1, 1, 0.1),
        colors='black',
        linewidths=1.5,
    )

    ax.clabel(cs, colors='black', fontsize=14)

    plt.show()


plot_score()


# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # H-like score

# %%
def plot_score():
    i2, i1 = np.mgrid[-0.8:0.8:50j, -0.8:0.8:50j]

    x = i1

    x_left = -0.5
    x_right = +0.5
    x_slope = 5
    assert x_left < x_right and x_slope > 0

    #    __
    # __/  \__
    x_score = np.maximum(
        -1,
        np.minimum(
            1, np.minimum(+x_slope * (x - x_left), -x_slope * (x - x_right))
        ),
    )

    slope_bin = 0.5 / 8
    bin_x = 0.2
    bin_y = 0.2
    min_x = -0.8
    y_left = -0.35
    y_right = +0.25
    y_slope = 5
    assert y_left < y_right and y_slope > 0
    y = i2 - (bin_y / bin_x * slope_bin) * (i1 - min_x)

    # __    __
    #   \__/
    y_score = np.minimum(
        1,
        np.maximum(
            -1, np.maximum(-y_slope * (y - y_left), +y_slope * (y - y_right))
        ),
    )

    score = np.minimum(x_score, y_score)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    ax.pcolormesh(
        i1,
        i2,
        score,
        cmap='RdBu',
    )
    ax.grid(True)
    plt.show()


plot_score()

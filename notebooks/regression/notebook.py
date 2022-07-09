# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% cellId="r60bke4u6ugchb8rkiobw"
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from joblib.memory import Memory
from sklearn import model_selection

from regression import cross_validation
from regression import histogram
from regression.load import build_df

PWD = Path(os.path.dirname(os.path.realpath('__file__')))
CACHE_STORAGE_PATH = PWD.joinpath('..', '..', '.storage', 'cache')
OUTPUT_STORAGE_PATH = PWD.joinpath(
    '..', '..', '..', 'sandbox', '.storage', 'output'
)

delay_to_style = {
    7: ':',
    30: '--',
    180: '-',
}

durations = ('4h', '3d', '24d')

delay_to_dir = {
    delay: OUTPUT_STORAGE_PATH.joinpath(
        'regression',
        'moex',
        'dohodru',
        'rub',
        '2015-07-01--2021-07-01',
        '_'.join(durations),
        'market_False',
        'profit_currency_USD',
        'cmf_False',
        'ad_False',
        'dln_True',
        'dln_no_vol_True',
        f'{delay}d',
    )
    for delay in (7, 30, 180)
}

delay_to_df = {
    delay: build_df(path, max_list_level=3)
    for delay, path in delay_to_dir.items()
}

time_series_split = model_selection.TimeSeriesSplit(n_splits=3)

memory_ = Memory(
    str(CACHE_STORAGE_PATH),
    mmap_mode='r',
    verbose=False,
)

DATE_RANGES = (
    (datetime(2015, 6, 1, 0, 0), datetime(2016, 6, 1, 0, 0)),
    (datetime(2016, 6, 1, 0, 0), datetime(2017, 6, 1, 0, 0)),
    (datetime(2017, 6, 1, 0, 0), datetime(2018, 6, 1, 0, 0)),
    (datetime(2018, 6, 1, 0, 0), datetime(2019, 6, 1, 0, 0)),
    (datetime(2019, 6, 1, 0, 0), datetime(2020, 6, 1, 0, 0)),
    (datetime(2020, 6, 1, 0, 0), datetime(2021, 6, 1, 0, 0)),
)

# %%
delay_to_df[180]

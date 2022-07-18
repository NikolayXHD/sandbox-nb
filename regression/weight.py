from __future__ import annotations

import typing

import numpy as np
import pandas as pd


def weighted_avg(w: pd.Series) -> typing.Callable[[pd.Series], np.ndarray]:
    def _agg(s: pd.Series) -> np.ndarray:
        return np.average(s, weights=w[s.index])

    return _agg


__all__ = ['weighted_avg']

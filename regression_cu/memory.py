from __future__ import annotations

import typing
import warnings

from joblib.memory import MemorizedFunc


def control_output(func: MemorizedFunc, verbose=False) -> typing.Callable:
    def decorated(*args, **kwargs):
        if verbose:
            is_cache_hit = func.check_call_in_cache(*args, **kwargs)
            print(f'{func.__name__} cache hit: {is_cache_hit}')
            return func(*args, **kwargs)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    r'^Persisting input arguments took \S+ to run\.',
                    UserWarning,
                )
                return func(*args, **kwargs)

    return decorated

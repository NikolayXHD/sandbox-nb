import inspect
import os

import importlib


def get_func_name(func, resolv_alias=True, win_characters=True):
    """Return the function import path (as a list of module names), and
    a name for the function.
    Parameters
    ----------
    func: callable
        The func to inspect
    resolv_alias: boolean, optional
        If true, possible local aliases are indicated.
    win_characters: boolean, optional
        If true, substitute special characters using urllib.quote
        This is useful in Windows, as it cannot encode some filenames
    """
    if hasattr(func, '__module__'):
        module = func.__module__
    else:
        try:
            module = inspect.getmodule(func)
        except TypeError:
            if hasattr(func, '__class__'):
                module = func.__class__.__module__
            else:
                module = 'unknown'
    if module is None:
        # Happens in doctests, eg
        module = ''
    if module == '__main__':
        try:
            filename = os.path.abspath(inspect.getsourcefile(func))
        except Exception:
            filename = None
        if filename is not None:
            # mangling of full path to filename
            parts = filename.split(os.sep)
            if parts[-1].startswith('<ipython-input'):
                # We're in a IPython (or notebook) session. parts[-1] comes
                # from func.__code__.co_filename and is of the form
                # <ipython-input-N-XYZ>, where:
                # - N is the cell number where the function was defined
                # - XYZ is a hash representing the function's code (and name).
                #   It will be consistent across sessions and kernel restarts,
                #   and will change if the function's code/name changes
                # We remove N so that cache is properly hit if the cell where
                # the func is defined is re-exectuted.
                # The XYZ hash should avoid collisions between functions with
                # the same name, both within the same notebook but also across
                # notebooks
                splitted = parts[-1].split('-')
                parts[-1] = '-'.join(splitted[:2] + splitted[3:])
            elif len(parts) > 2 and parts[-2].startswith('ipykernel_'):
                # In a notebook session (ipykernel). Filename seems to be 'xyz'
                # of above. parts[-2] has the structure ipykernel_XXXXXX where
                # XXXXXX is a six-digit number identifying the current run (?).
                # If we split it off, the function again has the same
                # identifier across runs.
                parts[-2] = 'ipykernel'
                del parts[-1]
            filename = '-'.join(parts)
            if filename.endswith('.py'):
                filename = filename[:-3]
            module = module + '-' + filename
    module = module.split('.')
    if hasattr(func, 'func_name'):
        name = func.func_name
    elif hasattr(func, '__name__'):
        name = func.__name__
    else:
        name = 'unknown'
    # Hack to detect functions not defined at the module-level
    if resolv_alias:
        # TODO: Maybe add a warning here?
        if hasattr(func, 'func_globals') and name in func.func_globals:
            if not func.func_globals[name] is func:
                name = '%s-alias' % name
    if inspect.ismethod(func):
        # We need to add the name of the class
        if hasattr(func, 'im_class'):
            klass = func.im_class
            module.append(klass.__name__)
    if os.name == 'nt' and win_characters:
        # Windows can't encode certain characters in filenames
        name = _clean_win_chars(name)
        module = [_clean_win_chars(s) for s in module]
    return module, name


try:
    import joblib.func_inspect
except ImportError:
    pass
else:
    from joblib.func_inspect import _clean_win_chars

    joblib.func_inspect.get_func_name = get_func_name

    import joblib

    importlib.reload(joblib)

    import joblib.memory

    importlib.reload(joblib.memory)

__all__ = []

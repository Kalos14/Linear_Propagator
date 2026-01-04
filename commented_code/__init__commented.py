"""Package convenience imports.

This is a very small helper module that:
- exposes everything from `propagator.py` at the package level, and
- provides a convenience function to reload submodules when working
  interactively (e.g., in an IPython session).

Note:
- The `reload(...)` call is Python 2 style. In Python 3 you typically need:
    import importlib
    importlib.reload(propagator)
"""

from propagator import *  # re-export functions/classes defined in propagator.py

def __reload_submodules__():
    """Reload submodules for interactive development.

    Useful when you are editing `propagator.py` and want to pick up changes
    without restarting the interpreter.
    """
    reload(propagator)

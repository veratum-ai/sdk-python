"""Backwards-compat shim: ``veratum.crosswalk`` → ``veratum.compliance.crosswalk``.

Note: ``veratum.compliance`` re-exports a function named ``crosswalk`` which
shadows the submodule attribute, so we use ``importlib`` to grab the real
submodule object and alias this module to it.
"""

import importlib as _importlib
import sys as _sys

_crosswalk_module = _importlib.import_module(".compliance.crosswalk", package="veratum")
_sys.modules[__name__] = _crosswalk_module

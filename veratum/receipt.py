"""Backwards-compat shim: ``veratum.receipt`` → ``veratum.core.receipt``."""

import sys as _sys
from .core import receipt as _receipt

_sys.modules[__name__] = _receipt

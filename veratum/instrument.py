"""Backwards-compat shim: ``veratum.instrument`` → ``veratum.core.instrument``."""

import sys as _sys
from .core import instrument as _instrument

_sys.modules[__name__] = _instrument

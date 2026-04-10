"""Backwards-compat shim: ``veratum.tiers`` → ``veratum.core.tiers``."""

import sys as _sys
from .core import tiers as _tiers

_sys.modules[__name__] = _tiers

"""Backwards-compat shim: ``veratum.evidence`` → ``veratum.core.evidence``."""

import sys as _sys
from .core import evidence as _evidence

_sys.modules[__name__] = _evidence

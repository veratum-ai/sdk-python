"""Backwards-compat shim: ``veratum.buffer`` → ``veratum.core.buffer``."""

import sys as _sys
from .core import buffer as _buffer

_sys.modules[__name__] = _buffer

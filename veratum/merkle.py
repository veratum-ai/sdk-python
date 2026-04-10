"""Backwards-compat shim: ``veratum.merkle`` → ``veratum.crypto.merkle``."""

import sys as _sys
from .crypto import merkle as _merkle

_sys.modules[__name__] = _merkle

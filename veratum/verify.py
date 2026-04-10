"""Backwards-compat shim: ``veratum.verify`` → ``veratum.crypto.verify``."""

import sys as _sys
from .crypto import verify as _verify

_sys.modules[__name__] = _verify

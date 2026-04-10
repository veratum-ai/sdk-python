"""Backwards-compat shim: ``veratum.transparency`` → ``veratum.crypto.transparency``."""

import sys as _sys
from .crypto import transparency as _transparency

_sys.modules[__name__] = _transparency

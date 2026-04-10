"""Backwards-compat shim: ``veratum.signing`` → ``veratum.crypto.signing``."""

import sys as _sys
from .crypto import signing as _signing

_sys.modules[__name__] = _signing

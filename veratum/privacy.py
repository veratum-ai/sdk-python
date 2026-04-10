"""Backwards-compat shim: ``veratum.privacy`` → ``veratum.security.privacy``."""

import sys as _sys
from .security import privacy as _privacy

_sys.modules[__name__] = _privacy

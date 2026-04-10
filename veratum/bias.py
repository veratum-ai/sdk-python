"""Backwards-compat shim: ``veratum.bias`` → ``veratum.compliance.bias``."""

import sys as _sys
from .compliance import bias as _bias

_sys.modules[__name__] = _bias

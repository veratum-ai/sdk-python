"""Backwards-compat shim: ``veratum.validation`` → ``veratum.compliance.validation``."""

import sys as _sys
from .compliance import validation as _validation

_sys.modules[__name__] = _validation

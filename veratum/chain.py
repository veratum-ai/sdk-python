"""Backwards-compat shim: ``veratum.chain`` → ``veratum.crypto.chain``.

The SDK was reorganized into ``veratum.crypto.*`` in the Schema 2.3.0 refactor.
This module preserves the legacy flat import path so that existing test suites
and downstream consumers continue to work without modification. We alias this
module to ``veratum.crypto.chain`` via ``sys.modules`` so even private
symbols (underscore-prefixed) remain accessible.
"""

import sys as _sys
from .crypto import chain as _chain

_sys.modules[__name__] = _chain

"""Core SDK — receipt creation, evidence engine, instrumentation.

Modules:
    sdk        — VeratumSDK main class and wrap() convenience function
    receipt    — Receipt data class (schema v2.1.0, 50+ compliance fields)
    evidence   — EvidenceEngine for universal evidence creation across integrations
    instrument — Auto-instrumentation hooks for LLM providers
    tiers      — Audit level presets (FULL, STANDARD, LIGHT, MINIMAL)
    buffer     — Resilient receipt buffer with retry and local persistence
"""

from .sdk import VeratumSDK, wrap
from .receipt import Receipt
from .evidence import EvidenceEngine, get_evidence_engine
from .instrument import Instrument
from .tiers import AuditLevel, get_audit_level, get_preset, list_presets, apply_preset
from .buffer import ReceiptBuffer

__all__ = [
    "VeratumSDK", "wrap",
    "Receipt",
    "EvidenceEngine", "get_evidence_engine",
    "Instrument",
    "AuditLevel", "get_audit_level", "get_preset", "list_presets", "apply_preset",
    "ReceiptBuffer",
]

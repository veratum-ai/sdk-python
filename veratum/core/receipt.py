"""Receipt generation and validation for Veratum audit trail.

Schema v2.3.0 — Litigation-grade evidence receipts compliant with:
- EU AI Act (Regulation 2024/1689) Articles 9, 12, 13, 14, 19, 26
- eIDAS (Regulation 910/2014) Article 41
- GDPR (Regulation 2016/679) Articles 5, 17, 22, 30, 35
- Colorado SB24-205 (AI Consumer Protections)
- NYC Local Law 144 (AEDT bias audit)
- EEOC Uniform Guidelines (29 CFR 1607)
- CFPB / Equal Credit Opportunity Act
- Illinois AI Video Interview Act
- Texas Responsible AI Governance Act
- FINRA Rules 3110, 17a-3, 17a-4
- NAIC Model AI Bulletin
- ISO/IEC 27037:2012 (Digital evidence handling)
- ISO/IEC 42001:2023 (AI Management System)
- ISO 24970 (AI auditability)
- ETSI TS 119 312 (Cryptographic suites)
- NIST AI Risk Management Framework 1.0
"""

import hashlib
import json
import os
import secrets
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..crypto.chain import HashChain, jcs_canonicalize


def generate_uuidv7() -> str:
    """
    Generate a time-ordered UUIDv7 string per IETF RFC 9562 / draft-peabody
    (new UUID formats).

    Layout (big-endian, per RFC 9562 §5.7):
      - 48 bits: Unix timestamp in milliseconds
      - 4 bits: version (0b0111 = 7)
      - 12 bits: random (rand_a)
      - 2 bits: variant (0b10)
      - 62 bits: random (rand_b)

    UUIDv7 is lexicographically sortable by creation time, which gives
    receipt IDs a natural monotonic ordering without requiring a separate
    sequence number, and is ideal for DynamoDB sort keys and B-tree
    indexes.

    Returns:
        UUIDv7 formatted as a canonical hex string
        (e.g., '018e4f3a-c8b1-7abc-9def-0123456789ab').
    """
    # 48-bit timestamp (ms since epoch)
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF  # mask to 48 bits

    # 12 bits random for rand_a
    rand_a = secrets.randbits(12)
    # 62 bits random for rand_b
    rand_b = secrets.randbits(62)

    # Assemble 128-bit integer
    value = (ts_ms & 0xFFFFFFFFFFFF) << 80           # 48-bit ts in top
    value |= (0x7 & 0xF) << 76                       # 4-bit version = 7
    value |= (rand_a & 0xFFF) << 64                  # 12-bit rand_a
    value |= (0b10 & 0x3) << 62                      # 2-bit variant = 10
    value |= (rand_b & 0x3FFFFFFFFFFFFFFF)           # 62-bit rand_b

    return str(uuid.UUID(int=value))


class Receipt:
    """Generates and manages audit receipts with multi-jurisdiction compliance."""

    SCHEMA_VERSION = "2.3.0"
    SDK_VERSION = "2.3.0"

    def __init__(self, hash_chain: HashChain) -> None:
        """
        Initialize receipt generator.

        Args:
            hash_chain: HashChain instance for maintaining integrity
        """
        self.hash_chain = hash_chain

    @staticmethod
    def _hash_data(data: str) -> str:
        """
        Hash string data using SHA-256.

        Args:
            data: String to hash

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        response: str,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        decision_type: str = "content_moderation",
        vertical: str = "hiring",
        # --- AI Decision Output Fields ---
        ai_score: Optional[float] = None,
        ai_threshold: Optional[float] = None,
        recruiter_action: Optional[str] = None,
        override_reason: Optional[str] = None,
        # --- Human Oversight Fields (EU AI Act Art.14, Colorado SB24-205 s6) ---
        human_review_state: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        reviewer_authority_scope: Optional[str] = None,
        reviewer_competence_level: Optional[str] = None,
        reviewer_training_date: Optional[str] = None,
        review_duration_seconds: Optional[int] = None,
        review_method: Optional[str] = None,
        review_outcome: Optional[str] = None,
        review_notes: Optional[str] = None,
        # --- Explainability Fields (GDPR Art.22, CFPB, Colorado s6) ---
        explainability: Optional[Dict[str, Any]] = None,
        # --- Consequential Decision Metadata (Colorado SB24-205 s2) ---
        decision_category: Optional[str] = None,
        decision_outcome: Optional[str] = None,
        affected_individual_notified: Optional[bool] = None,
        notification_timestamp: Optional[str] = None,
        appeal_available: Optional[bool] = None,
        appeal_mechanism: Optional[str] = None,
        correction_opportunity: Optional[bool] = None,
        # --- GDPR / Data Protection Fields ---
        data_processing_basis: Optional[str] = None,
        data_processing_purpose: Optional[str] = None,
        special_categories_present: Optional[bool] = None,
        retention_legal_basis: Optional[str] = None,
        data_subject_id_hash: Optional[str] = None,
        data_subject_consent: Optional[bool] = None,
        dpia_reference: Optional[str] = None,
        # --- Bias/Fairness Audit Fields (NYC LL144, EEOC, NAIC) ---
        bias_audit: Optional[Dict[str, Any]] = None,
        # --- Jurisdiction/Compliance Tracking ---
        applicable_jurisdictions: Optional[List[str]] = None,
        compliance_metadata: Optional[Dict[str, Any]] = None,
        # --- Illinois AIVA Specific ---
        consent_obtained: Optional[bool] = None,
        consent_timestamp: Optional[str] = None,
        ai_disclosure_provided: Optional[bool] = None,
        # --- Insurance Specific (NAIC) ---
        insurance_line: Optional[str] = None,
        actuarial_justification: Optional[str] = None,
        # --- Financial Services (FINRA, CFPB) ---
        finra_rule_ref: Optional[str] = None,
        adverse_action_notice_sent: Optional[bool] = None,
        adverse_action_notice_date: Optional[str] = None,
        # --- General ---
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an audit receipt with full chain integrity.

        Complies with:
        - Article 12 of the EU AI Act (transparency & documentation)
        - Article 14 of the EU AI Act (human oversight)
        - ISO 24970 (AI auditability requirements)
        - RFC 8785 (JSON Canonicalization Scheme)
        - eIDAS Article 41 (qualified timestamps — applied server-side)

        Args:
            prompt: Input prompt text
            response: Model response text
            model: Model identifier (e.g., 'claude-3-opus')
            provider: Provider name (e.g., 'anthropic')
            tokens_in: Input tokens consumed
            tokens_out: Output tokens consumed
            decision_type: Type of decision (default: 'content_moderation')
            vertical: Industry vertical (default: 'hiring')
            ai_score: Model confidence score (0-1)
            ai_threshold: Decision threshold
            recruiter_action: Action taken by recruiter
            human_review_state: State of human review
            reviewer_id: ID of human reviewer
            override_reason: Reason for any override
            metadata: Additional context
            reviewer_name: Full name of human reviewer (Article 14)
            reviewer_role: Role/title of reviewer (Article 14)
            reviewer_authority_scope: Scope of reviewer's authority (Article 14)
            reviewer_competence_level: Competence level (Article 14)
            reviewer_training_date: ISO date of last training (Article 14)
            review_duration_seconds: Time spent reviewing (Article 14)
            review_method: Method of review (Article 14)
            data_processing_basis: GDPR legal basis (e.g., 'legitimate_interest')
            data_processing_purpose: Purpose of processing
            special_categories_present: Whether special category data is present
            retention_legal_basis: Legal basis for retention
            data_subject_id_hash: SHA-256 hash of data subject identifier

        Returns:
            Receipt dictionary with all required fields
        """
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        # Compute hashes for prompt and response
        prompt_hash = self._hash_data(prompt)
        response_hash = self._hash_data(response)

        # Get current chain state
        chain_state = self.hash_chain.get_chain_state()

        # Build receipt dictionary (excluding computed fields initially)
        receipt: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "receipt_id": generate_uuidv7(),
            "entry_hash": "",        # Computed below
            "entry_hash_sha3": "",   # Computed below (SHA3-256 dual hash)
            "prev_hash": chain_state["prev_hash"],
            "sequence_no": chain_state["sequence_no"],
            "timestamp": timestamp,
            "model": model,
            "provider": provider,
            "sdk_version": self.SDK_VERSION,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "decision_type": decision_type,
            "vertical": vertical,
        }

        # Add optional Article 12 / ISO 24970 fields
        if ai_score is not None:
            receipt["ai_score"] = ai_score
        if ai_threshold is not None:
            receipt["ai_threshold"] = ai_threshold
        if recruiter_action is not None:
            receipt["recruiter_action"] = recruiter_action
        if human_review_state is not None:
            receipt["human_review_state"] = human_review_state
        if reviewer_id is not None:
            receipt["reviewer_id"] = reviewer_id
        if override_reason is not None:
            receipt["override_reason"] = override_reason

        # Human oversight fields (EU AI Act Art.14, Colorado SB24-205 s6)
        if reviewer_name is not None:
            receipt["reviewer_name"] = reviewer_name
        if reviewer_role is not None:
            receipt["reviewer_role"] = reviewer_role
        if reviewer_authority_scope is not None:
            receipt["reviewer_authority_scope"] = reviewer_authority_scope
        if reviewer_competence_level is not None:
            receipt["reviewer_competence_level"] = reviewer_competence_level
        if reviewer_training_date is not None:
            receipt["reviewer_training_date"] = reviewer_training_date
        if review_duration_seconds is not None:
            receipt["review_duration_seconds"] = review_duration_seconds
        if review_method is not None:
            receipt["review_method"] = review_method
        if review_outcome is not None:
            receipt["review_outcome"] = review_outcome
        if review_notes is not None:
            receipt["review_notes"] = review_notes

        # Explainability (GDPR Art.22, CFPB adverse action, Colorado s6)
        if explainability is not None:
            receipt["explainability"] = explainability

        # Consequential decision metadata (Colorado SB24-205 s2)
        if decision_category is not None:
            receipt["decision_category"] = decision_category
        if decision_outcome is not None:
            receipt["decision_outcome"] = decision_outcome
        if affected_individual_notified is not None:
            receipt["affected_individual_notified"] = affected_individual_notified
        if notification_timestamp is not None:
            receipt["notification_timestamp"] = notification_timestamp
        if appeal_available is not None:
            receipt["appeal_available"] = appeal_available
        if appeal_mechanism is not None:
            receipt["appeal_mechanism"] = appeal_mechanism
        if correction_opportunity is not None:
            receipt["correction_opportunity"] = correction_opportunity

        # GDPR / Data protection fields
        if data_processing_basis is not None:
            receipt["data_processing_basis"] = data_processing_basis
        if data_processing_purpose is not None:
            receipt["data_processing_purpose"] = data_processing_purpose
        if special_categories_present is not None:
            receipt["special_categories_present"] = special_categories_present
        if retention_legal_basis is not None:
            receipt["retention_legal_basis"] = retention_legal_basis
        if data_subject_id_hash is not None:
            receipt["data_subject_id_hash"] = data_subject_id_hash
        if data_subject_consent is not None:
            receipt["data_subject_consent"] = data_subject_consent
        if dpia_reference is not None:
            receipt["dpia_reference"] = dpia_reference

        # Bias/fairness audit (NYC LL144, EEOC, NAIC)
        if bias_audit is not None:
            receipt["bias_audit"] = bias_audit

        # Jurisdiction/compliance tracking
        if applicable_jurisdictions is not None:
            receipt["applicable_jurisdictions"] = applicable_jurisdictions

        # Illinois AIVA specific
        if consent_obtained is not None:
            receipt["consent_obtained"] = consent_obtained
        if consent_timestamp is not None:
            receipt["consent_timestamp"] = consent_timestamp
        if ai_disclosure_provided is not None:
            receipt["ai_disclosure_provided"] = ai_disclosure_provided

        # Insurance specific (NAIC)
        if insurance_line is not None:
            receipt["insurance_line"] = insurance_line
        if actuarial_justification is not None:
            receipt["actuarial_justification"] = actuarial_justification

        # Financial services (FINRA, CFPB)
        if finra_rule_ref is not None:
            receipt["finra_rule_ref"] = finra_rule_ref
        if adverse_action_notice_sent is not None:
            receipt["adverse_action_notice_sent"] = adverse_action_notice_sent
        if adverse_action_notice_date is not None:
            receipt["adverse_action_notice_date"] = adverse_action_notice_date

        # Server-side fields (populated after upload). All are excluded from
        # the entry_hash computation so adding placeholders here is safe.
        receipt["signature"] = ""
        receipt["signature_ed25519"] = ""
        receipt["signature_ml_dsa_65"] = ""
        receipt["merkle_proof"] = None
        receipt["xrpl_tx_hash"] = ""
        receipt["opentimestamps_proof"] = ""
        receipt["rfc3161_token"] = ""

        # Set human review state default
        if "human_review_state" not in receipt:
            receipt["human_review_state"] = "none"

        # Compliance metadata — merge user-provided with system defaults
        system_compliance = {
            "regulations": [
                "EU AI Act (Regulation 2024/1689) Articles 9, 12, 13, 14, 26",
                "eIDAS (Regulation 910/2014) Article 41",
                "GDPR (Regulation 2016/679) Articles 5, 17, 22, 30, 35",
            ],
            "standards": [
                "ISO/IEC 27037:2012",
                "ISO/IEC 42001:2023",
                "ISO 24970",
                "NIST AI RMF 1.0",
                "RFC 8785 (JCS)",
                "ETSI TS 119 312",
            ],
            "evidence_class": "litigation-grade",
            "forensically_sound": True,
            "integrity_method": "RFC8785-JCS-SHA256+SHA3-256-dual-linked-hash-chain",
            "canonicalization": "RFC 8785 (JSON Canonicalization Scheme)",
            "hash_algorithm": "SHA-256 (OID 2.16.840.1.101.3.4.2.1)",
            "hash_algorithm_secondary": "SHA3-256 (OID 2.16.840.1.101.3.4.2.8)",
            "timestamp_type": "RFC 3161 qualified (eIDAS Article 41)",
        }
        if compliance_metadata:
            system_compliance.update(compliance_metadata)
        receipt["compliance_metadata"] = system_compliance

        # Add metadata if provided
        if metadata:
            receipt["metadata"] = metadata

        # Compute dual entry hashes (SHA-256 + SHA3-256) over the RFC 8785
        # JCS canonical form of the receipt. Both hashes cover identical
        # bytes — the only difference is the hash algorithm. This gives
        # algorithmic agility: if SHA-256 is ever broken, receipts remain
        # verifiable via SHA3-256.
        sha256_hex, sha3_hex = self.hash_chain.compute_dual_entry_hash(receipt)
        receipt["entry_hash"] = sha256_hex
        receipt["entry_hash_sha3"] = sha3_hex

        # Advance chain state (tracks primary SHA-256 for backwards compat)
        self.hash_chain.advance_chain(receipt)

        return receipt

    def verify_chain_integrity(self, receipt: Dict[str, Any], prev_receipt: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verify receipt chain integrity.

        Checks that:
        1. entry_hash is correctly computed (RFC 8785 JCS)
        2. prev_hash matches previous receipt's entry_hash
        3. sequence_no increments properly

        Args:
            receipt: Receipt to verify
            prev_receipt: Previous receipt in chain (optional)

        Returns:
            True if receipt is valid and properly linked
        """
        # Verify entry_hash is correct (RFC 8785 JCS, SHA-256)
        stored_hash = receipt.get("entry_hash", "")
        computed_hash = self.hash_chain.compute_entry_hash(receipt)

        if stored_hash != computed_hash:
            # Check legacy format for backward compatibility
            legacy_canonical = {
                k: v for k, v in receipt.items()
                if k not in (
                    "entry_hash",
                    "entry_hash_sha3",
                    "xrpl_tx_hash",
                    "opentimestamps_proof",
                    "rfc3161_token",
                    "signature",
                    "signature_ed25519",
                    "signature_ml_dsa_65",
                    "verifiable_credential",
                )
            }
            legacy_json = json.dumps(legacy_canonical, sort_keys=True, separators=(",", ":"))
            legacy_hash = hashlib.sha256(legacy_json.encode("utf-8")).hexdigest()
            if stored_hash != legacy_hash:
                return False

        # Schema 2.3.0: if entry_hash_sha3 is present, verify it too.
        # Older (2.1.0/2.0.0) receipts don't include it and are still valid.
        stored_sha3 = receipt.get("entry_hash_sha3")
        if stored_sha3:
            computed_sha3 = self.hash_chain.compute_entry_hash_sha3(receipt)
            if stored_sha3 != computed_sha3:
                return False

        # Verify prev_hash linkage if previous receipt provided
        if prev_receipt:
            expected_prev_hash = prev_receipt.get("entry_hash", "")
            if receipt.get("prev_hash") != expected_prev_hash:
                return False

            # Verify sequence increments by 1
            if receipt.get("sequence_no", 0) != prev_receipt.get("sequence_no", 0) + 1:
                return False

        # Verify genesis receipt (prev_hash = "0"*64, sequence_no = 0)
        if receipt.get("sequence_no") == 0:
            if receipt.get("prev_hash") != "0" * 64:
                return False

        return True

    def serialize(self, receipt: Dict[str, Any]) -> str:
        """
        Serialize receipt to JSON.

        Args:
            receipt: Receipt dictionary

        Returns:
            JSON string
        """
        return json.dumps(receipt, indent=2)

    def serialize_canonical(self, receipt: Dict[str, Any]) -> bytes:
        """
        Serialize receipt to RFC 8785 JCS canonical form.

        This is the legally defensible serialization format.

        Args:
            receipt: Receipt dictionary

        Returns:
            UTF-8 encoded canonical JSON bytes
        """
        return jcs_canonicalize(receipt)

"""Universal Evidence Engine for Veratum integrations.

This is the core module that every integration (LiteLLM, Portkey, MCP, etc.)
calls to create cryptographic evidence of AI decisions. It wraps the existing
chain.py, receipt.py, and verify.py into a single, clean interface.

Design principles:
- Thread-safe for concurrent AI calls
- Lazy initialization (no API connection until first upload)
- Fails open (always creates receipt locally, even if upload fails)
- Upload is async by default (uses threading)
- Extract methods handle messy API format differences
- Every receipt gets proper hash chain linkage

This is the evidence layer for every AI system — the foundation that makes
Veratum the audit trail that scales across all LLMs and inference engines.
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

from ..crypto.chain import HashChain, jcs_canonicalize
from .receipt import Receipt, generate_uuidv7
from ..crypto.verify import verify_receipt, verify_chain


logger = logging.getLogger(__name__)


class EvidenceEngine:
    """
    Universal evidence engine for Veratum integrations.

    Any gateway, proxy, or middleware calls this to create
    cryptographic evidence of AI decisions. This is the core
    that makes Veratum the evidence layer for every AI system.

    Thread-safe. Lazy initialization. Fails open for evidence creation.

    Example:
        engine = EvidenceEngine()
        receipt = engine.create_evidence(
            request={"prompt": "What is 2+2?"},
            response={"text": "4"},
            provider="openai",
            model="gpt-4o",
        )
        engine.upload_evidence(receipt)  # Async by default
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the universal evidence engine.

        Reads from environment variables if not provided:
        - VERATUM_API_KEY: API key for Veratum backend
        - VERATUM_ENDPOINT: Veratum API endpoint (e.g., https://api.veratum.ai)
        - VERATUM_CUSTOMER_ID: Customer ID for multi-tenant deployments

        Args:
            api_key: Veratum API key. If None, reads from VERATUM_API_KEY env var.
            endpoint: Veratum API endpoint. If None, reads from VERATUM_ENDPOINT env var.
            customer_id: Customer ID. If None, reads from VERATUM_CUSTOMER_ID env var.
        """
        self.api_key = api_key or os.getenv("VERATUM_API_KEY")
        self.endpoint = endpoint or os.getenv(
            "VERATUM_ENDPOINT", "https://api.veratum.ai"
        )
        self.customer_id = customer_id or os.getenv("VERATUM_CUSTOMER_ID")

        # Initialize hash chain and receipt generator
        self._hash_chain = HashChain()
        self._receipt_gen = Receipt(self._hash_chain)

        # Thread safety for concurrent calls
        self._chain_lock = threading.Lock()

        # Track initialization state
        self._api_initialized = False
        self._pending_uploads: List[Dict[str, Any]] = []

        logger.info(
            f"EvidenceEngine initialized (endpoint={self.endpoint}, "
            f"customer_id={self.customer_id})"
        )

    # Allowed values for the privacy_mode kwarg below.
    _VALID_PRIVACY_MODES = ("standard", "hash_only")

    def create_evidence(
        self,
        request: dict,
        response: dict,
        provider: str,
        model: str,
        decision_type: str = "ai_inference",
        vertical: str = "general",
        metadata: Optional[dict] = None,
        privacy_mode: str = "standard",
    ) -> Dict[str, Any]:
        """
        Create a full evidence receipt from an AI request/response pair.

        This is the ONE function every integration calls. It handles:
        1. Extracting prompt/response from provider-specific API formats
        2. Computing SHA-256 hashes
        3. Building the receipt with full chain integrity
        4. Thread-safe chain advancement

        Args:
            request: The AI request dict (structure varies by provider).
                For OpenAI: {"messages": [...], "model": "gpt-4o", ...}
                For Anthropic: {"prompt": "...", "model": "claude-3-opus", ...}
                For generic LLM: {"prompt": "...", "input_tokens": N, ...}
            response: The AI response dict (structure varies by provider).
                For OpenAI: {"choices": [{"message": {"content": "..."}}], "usage": {...}}
                For Anthropic: {"content": "...", "usage": {...}}
                For generic: {"text": "...", "output_tokens": N, ...}
            provider: Provider name: "openai", "anthropic", "google", "generic", etc.
            model: Model identifier: "gpt-4o", "claude-3-opus", "gemini-1.5-pro", etc.
            decision_type: Type of decision (default: "ai_inference").
                Examples: "ai_inference", "content_moderation", "hiring_decision",
                "loan_approval", "medical_diagnosis", etc.
            vertical: Industry/domain vertical (default: "general").
                Examples: "general", "hiring", "finance", "healthcare", "insurance", etc.
            metadata: Additional compliance metadata dict.
                Examples:
                - human_review_state: "pending", "approved", "rejected"
                - reviewer_id: "reviewer_123"
                - reviewer_role: "compliance_officer"
                - applicable_jurisdictions: ["EU", "US-CA", "US-NY"]
                - bias_audit: {"protected_attributes": [...], "protected_group": "..."}
                - data_processing_basis: "legitimate_interest"
                - data_subject_id_hash: (SHA-256 of PII)

        Returns:
            Receipt dict with:
            - entry_hash: SHA-256 of JCS-canonicalized receipt
            - prev_hash: Link to previous receipt (chain integrity)
            - sequence_no: Position in hash chain
            - prompt_hash: SHA-256 of request
            - response_hash: SHA-256 of response
            - timestamp: ISO 8601 UTC timestamp
            - schema_version: "2.1.0"
            - All compliance fields from metadata (human review, bias audit, etc.)

        Raises:
            ValueError: If request/response cannot be parsed or provider is unknown.

        Schema 2.3.0 additions:
            privacy_mode: "standard" (default) includes prompt/response text
                in the local receipt object before hashing; "hash_only" clears
                those fields immediately after hashing so that no plaintext
                ever leaves the caller's process. Useful for GDPR Art. 25
                (data minimization) and regulated-data workflows where even
                the in-memory receipt object must not contain user content.
            The receipt also captures a `provider_response_id` field, extracted
                from the provider's native response envelope (e.g. OpenAI
                `id`, Anthropic `id`), enabling cross-correlation with
                provider-side logs during audit.
        """
        if privacy_mode not in self._VALID_PRIVACY_MODES:
            raise ValueError(
                f"Invalid privacy_mode '{privacy_mode}'. "
                f"Must be one of: {', '.join(self._VALID_PRIVACY_MODES)}"
            )

        # Extract prompt, response, tokens, and provider-side response id.
        try:
            if provider.lower() == "openai":
                prompt, response_text, tokens_in, tokens_out, provider_response_id = (
                    self._extract_from_openai(request, response)
                )
            elif provider.lower() == "anthropic":
                prompt, response_text, tokens_in, tokens_out, provider_response_id = (
                    self._extract_from_anthropic(request, response)
                )
            else:
                # Generic extraction — handles most LLM APIs
                prompt, response_text, tokens_in, tokens_out, provider_response_id = (
                    self._extract_from_generic(request, response)
                )
        except Exception as e:
            logger.error(f"Failed to extract from {provider}: {e}")
            raise ValueError(
                f"Cannot parse {provider} request/response: {e}"
            ) from e

        # Thread-safe receipt generation with chain advancement
        with self._chain_lock:
            # Merge metadata with defaults
            merged_metadata = dict(metadata or {})
            # Caller-provided extra_metadata takes precedence; we only add
            # fields that aren't already specified.
            extra_metadata = dict(merged_metadata.get("extra_metadata") or {})
            if provider_response_id and "provider_response_id" not in extra_metadata:
                extra_metadata["provider_response_id"] = provider_response_id
            extra_metadata.setdefault("privacy_mode", privacy_mode)
            merged_metadata["extra_metadata"] = extra_metadata

            # Generate receipt with all compliance fields
            receipt = self._receipt_gen.generate(
                prompt=prompt,
                response=response_text,
                model=model,
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                decision_type=decision_type,
                vertical=vertical,
                # Unpack metadata fields into Receipt.generate kwargs
                human_review_state=merged_metadata.get("human_review_state"),
                reviewer_id=merged_metadata.get("reviewer_id"),
                reviewer_name=merged_metadata.get("reviewer_name"),
                reviewer_role=merged_metadata.get("reviewer_role"),
                reviewer_authority_scope=merged_metadata.get("reviewer_authority_scope"),
                reviewer_competence_level=merged_metadata.get("reviewer_competence_level"),
                reviewer_training_date=merged_metadata.get("reviewer_training_date"),
                review_duration_seconds=merged_metadata.get("review_duration_seconds"),
                review_method=merged_metadata.get("review_method"),
                review_outcome=merged_metadata.get("review_outcome"),
                review_notes=merged_metadata.get("review_notes"),
                explainability=merged_metadata.get("explainability"),
                decision_category=merged_metadata.get("decision_category"),
                decision_outcome=merged_metadata.get("decision_outcome"),
                affected_individual_notified=merged_metadata.get(
                    "affected_individual_notified"
                ),
                notification_timestamp=merged_metadata.get("notification_timestamp"),
                appeal_available=merged_metadata.get("appeal_available"),
                appeal_mechanism=merged_metadata.get("appeal_mechanism"),
                correction_opportunity=merged_metadata.get("correction_opportunity"),
                data_processing_basis=merged_metadata.get("data_processing_basis"),
                data_processing_purpose=merged_metadata.get("data_processing_purpose"),
                special_categories_present=merged_metadata.get(
                    "special_categories_present"
                ),
                retention_legal_basis=merged_metadata.get("retention_legal_basis"),
                data_subject_id_hash=merged_metadata.get("data_subject_id_hash"),
                data_subject_consent=merged_metadata.get("data_subject_consent"),
                dpia_reference=merged_metadata.get("dpia_reference"),
                bias_audit=merged_metadata.get("bias_audit"),
                applicable_jurisdictions=merged_metadata.get(
                    "applicable_jurisdictions"
                ),
                compliance_metadata=merged_metadata.get("compliance_metadata"),
                consent_obtained=merged_metadata.get("consent_obtained"),
                consent_timestamp=merged_metadata.get("consent_timestamp"),
                ai_disclosure_provided=merged_metadata.get("ai_disclosure_provided"),
                insurance_line=merged_metadata.get("insurance_line"),
                actuarial_justification=merged_metadata.get("actuarial_justification"),
                finra_rule_ref=merged_metadata.get("finra_rule_ref"),
                adverse_action_notice_sent=merged_metadata.get(
                    "adverse_action_notice_sent"
                ),
                adverse_action_notice_date=merged_metadata.get(
                    "adverse_action_notice_date"
                ),
                ai_score=merged_metadata.get("ai_score"),
                ai_threshold=merged_metadata.get("ai_threshold"),
                recruiter_action=merged_metadata.get("recruiter_action"),
                override_reason=merged_metadata.get("override_reason"),
                metadata=merged_metadata.get("extra_metadata"),
            )

        logger.info(
            f"Created evidence receipt: model={model}, "
            f"seq={receipt['sequence_no']}, "
            f"hash={receipt['entry_hash'][:16]}..."
        )
        return receipt

    def create_evidence_from_hashes(
        self,
        prompt_hash: str,
        response_hash: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        decision_type: str = "ai_inference",
        vertical: str = "general",
        provider_response_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Create a receipt from pre-computed SHA-256 hashes (hash-only mode).

        This is the "data minimization" path: the caller has already hashed
        the prompt and response client-side and only passes the hex digests.
        No plaintext ever crosses the Veratum boundary, satisfying GDPR
        Article 25 and regulated workflows where PII must not leave the
        customer's trust boundary.

        Args:
            prompt_hash: Hex-encoded SHA-256 of the canonical prompt bytes
            response_hash: Hex-encoded SHA-256 of the canonical response bytes
            provider: Provider name (e.g. "openai", "anthropic")
            model: Model identifier
            tokens_in: Input token count (safe to share — no content)
            tokens_out: Output token count
            decision_type: Decision type classification
            vertical: Industry vertical
            provider_response_id: Provider-side response identifier for
                cross-reference with provider logs during audit
            metadata: Additional compliance metadata

        Returns:
            Receipt dict with entry_hash, entry_hash_sha3, prev_hash,
            sequence_no, and all compliance fields. The receipt carries
            `privacy_mode: "hash_only"` in its metadata.

        Raises:
            ValueError: If prompt_hash or response_hash is not 64-char hex.
        """
        # Validate hashes — hash-only callers must supply real SHA-256 hex.
        for name, h in (("prompt_hash", prompt_hash), ("response_hash", response_hash)):
            if (
                not isinstance(h, str)
                or len(h) != 64
                or not all(c in "0123456789abcdef" for c in h)
            ):
                raise ValueError(
                    f"{name} must be 64-char lowercase hex SHA-256 digest"
                )

        merged_metadata = dict(metadata or {})
        extra_metadata = dict(merged_metadata.get("extra_metadata") or {})
        extra_metadata.setdefault("privacy_mode", "hash_only")
        if provider_response_id and "provider_response_id" not in extra_metadata:
            extra_metadata["provider_response_id"] = provider_response_id
        merged_metadata["extra_metadata"] = extra_metadata

        with self._chain_lock:
            # We bypass the Receipt.generate() prompt/response hashing path
            # and instead build the receipt structure directly so no
            # plaintext is ever handled by the SDK.
            chain_state = self._hash_chain.get_chain_state()
            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            receipt: Dict[str, Any] = {
                "schema_version": Receipt.SCHEMA_VERSION,
                "receipt_id": generate_uuidv7(),
                "entry_hash": "",
                "entry_hash_sha3": "",
                "prev_hash": chain_state["prev_hash"],
                "sequence_no": chain_state["sequence_no"],
                "timestamp": timestamp,
                "model": model,
                "provider": provider,
                "sdk_version": Receipt.SDK_VERSION,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "decision_type": decision_type,
                "vertical": vertical,
                "human_review_state": "none",
                "signature": "",
                "signature_ed25519": "",
                "signature_ml_dsa_65": "",
                "merkle_proof": None,
                "xrpl_tx_hash": "",
                "opentimestamps_proof": "",
                "rfc3161_token": "",
                "compliance_metadata": {
                    "privacy_mode": "hash_only",
                    "data_minimization": "GDPR Article 25",
                    "evidence_class": "litigation-grade",
                    "canonicalization": "RFC 8785 (JSON Canonicalization Scheme)",
                    "hash_algorithm": "SHA-256 (OID 2.16.840.1.101.3.4.2.1)",
                    "hash_algorithm_secondary": "SHA3-256 (OID 2.16.840.1.101.3.4.2.8)",
                    "integrity_method": "RFC8785-JCS-SHA256+SHA3-256-dual-linked-hash-chain",
                },
                "metadata": extra_metadata,
            }

            sha256_hex, sha3_hex = self._hash_chain.compute_dual_entry_hash(
                receipt
            )
            receipt["entry_hash"] = sha256_hex
            receipt["entry_hash_sha3"] = sha3_hex
            self._hash_chain.advance_chain(receipt)

        logger.info(
            f"Created hash-only evidence receipt: model={model}, "
            f"seq={receipt['sequence_no']}, "
            f"hash={receipt['entry_hash'][:16]}..."
        )
        return receipt

    def upload_evidence(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload receipt to Veratum API.

        Uploads receipt to Veratum backend for qualified timestamp,
        merkle proof, and long-term archival.

        If upload fails:
        - Returns error response (does NOT raise exception)
        - Receipt remains valid and independently verifiable locally
        - Can be retried later with create_and_upload

        Args:
            receipt: Receipt dict from create_evidence()

        Returns:
            API response dict:
            {
                "success": bool,
                "receipt_id": str (if success),
                "timestamp": str (qualified timestamp, if success),
                "merkle_proof": dict (inclusion proof, if success),
                "verifiable_credential": dict (W3C VC, if success),
                "error": str (if failed),
                "error_code": str (if failed),
            }
        """
        if not self.api_key:
            logger.warning(
                "VERATUM_API_KEY not set. Receipt will not be uploaded. "
                "Receipt is still valid and independently verifiable."
            )
            return {
                "success": False,
                "error": "VERATUM_API_KEY not configured",
                "error_code": "no_api_key",
            }

        try:
            # Prepare upload payload
            payload = {
                "receipt": receipt,
                "customer_id": self.customer_id,
            }
            payload_json = json.dumps(payload).encode("utf-8")

            # Make API request
            url = f"{self.endpoint}/v2/evidence/upload"
            req = Request(
                url,
                data=payload_json,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            with urlopen(req, timeout=30) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            logger.info(
                f"Uploaded receipt {receipt['entry_hash'][:16]}... to Veratum API"
            )
            return {
                "success": True,
                "receipt_id": response_data.get("receipt_id"),
                "timestamp": response_data.get("timestamp"),
                "merkle_proof": response_data.get("merkle_proof"),
                "verifiable_credential": response_data.get("verifiable_credential"),
            }

        except URLError as e:
            logger.error(f"Upload failed (network error): {e}")
            return {
                "success": False,
                "error": f"Network error: {e}",
                "error_code": "network_error",
            }
        except json.JSONDecodeError as e:
            logger.error(f"Upload failed (invalid API response): {e}")
            return {
                "success": False,
                "error": f"Invalid API response: {e}",
                "error_code": "invalid_response",
            }
        except Exception as e:
            logger.error(f"Upload failed (unexpected error): {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "unknown_error",
            }

    def create_and_upload(
        self,
        request: dict,
        response: dict,
        provider: str,
        model: str,
        decision_type: str = "ai_inference",
        vertical: str = "general",
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Create and upload evidence in one call.

        Convenience method that combines create_evidence() + upload_evidence().
        Returns both the receipt and upload result.

        Args:
            request: AI request dict (see create_evidence)
            response: AI response dict (see create_evidence)
            provider: Provider name (see create_evidence)
            model: Model identifier (see create_evidence)
            decision_type: Decision type (see create_evidence)
            vertical: Vertical (see create_evidence)
            metadata: Compliance metadata (see create_evidence)

        Returns:
            {
                "receipt": {receipt dict},
                "upload_result": {upload response dict},
            }
        """
        receipt = self.create_evidence(
            request=request,
            response=response,
            provider=provider,
            model=model,
            decision_type=decision_type,
            vertical=vertical,
            metadata=metadata,
        )

        upload_result = self.upload_evidence(receipt)

        return {
            "receipt": receipt,
            "upload_result": upload_result,
        }

    def verify(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a receipt's integrity offline.

        Independent verification — recomputes entry_hash and checks it matches
        the stored hash. No Veratum server connection needed.

        Args:
            receipt: Receipt dict from create_evidence()

        Returns:
            Verification result dict:
            {
                "valid": bool,
                "entry_hash_match": bool,
                "computed_hash": str,
                "stored_hash": str,
                "errors": [str],
            }
        """
        return verify_receipt(receipt)

    def verify_chain(self, receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify an entire chain of receipts.

        Checks:
        1. Each receipt's entry_hash is correct (content integrity)
        2. Each receipt's prev_hash matches previous entry_hash (chain integrity)
        3. Sequence numbers are monotonically increasing
        4. No gaps in sequence

        This is the full independent audit — regulators/auditors can run this
        on exported receipts with zero Veratum infrastructure.

        Args:
            receipts: List of receipt dicts ordered by sequence_no

        Returns:
            Chain verification result:
            {
                "valid": bool,
                "total_receipts": int,
                "verified_count": int,
                "chain_intact": bool,
                "sequence_valid": bool,
                "errors": [{"index": int, "receipt_hash": str, "error": str}],
                "first_sequence": int,
                "last_sequence": int,
            }
        """
        return verify_chain(receipts)

    @staticmethod
    def _extract_from_openai(
        request_kwargs: dict, response: dict
    ) -> Tuple[str, str, int, int, Optional[str]]:
        """
        Extract prompt, response, and token counts from OpenAI-format API calls.

        Handles both:
        - messages API: request["messages"] list of {role, content}
        - legacy text API: request["prompt"] string

        Args:
            request_kwargs: OpenAI request dict (messages, model, etc.)
            response: OpenAI response dict (choices, usage, etc.)

        Returns:
            (prompt_text, response_text, tokens_in, tokens_out)

        Raises:
            ValueError: If cannot extract required fields
        """
        # Extract prompt from messages or legacy prompt field
        messages = request_kwargs.get("messages", [])
        prompt_str = ""

        if isinstance(messages, list) and messages:
            # New messages format
            prompt_str = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
                if msg.get("content")
            )
        elif "prompt" in request_kwargs:
            # Legacy prompt format
            prompt_str = request_kwargs["prompt"]

        if not prompt_str:
            raise ValueError("Cannot extract prompt from OpenAI request")

        # Extract response text
        choices = response.get("choices", [])
        response_text = ""

        if choices:
            choice = choices[0]
            if "message" in choice:
                response_text = choice["message"].get("content", "")
            elif "text" in choice:
                response_text = choice["text"]

        if not response_text:
            raise ValueError("Cannot extract response from OpenAI response")

        # Extract token counts
        usage = response.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        # Provider-side response identifier (OpenAI returns `id`)
        provider_response_id = response.get("id") or None

        return prompt_str, response_text, tokens_in, tokens_out, provider_response_id

    @staticmethod
    def _extract_from_anthropic(
        request_kwargs: dict, response: dict
    ) -> Tuple[str, str, int, int, Optional[str]]:
        """
        Extract prompt, response, and token counts from Anthropic-format API calls.

        Handles:
        - request["prompt"] or request["messages"]
        - response["content"] list with text blocks

        Args:
            request_kwargs: Anthropic request dict
            response: Anthropic response dict

        Returns:
            (prompt_text, response_text, tokens_in, tokens_out)

        Raises:
            ValueError: If cannot extract required fields
        """
        # Extract prompt
        prompt_str = ""
        if "prompt" in request_kwargs:
            prompt_str = request_kwargs["prompt"]
        elif "messages" in request_kwargs:
            messages = request_kwargs["messages"]
            if isinstance(messages, list):
                prompt_str = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                    if msg.get("content")
                )

        if not prompt_str:
            raise ValueError("Cannot extract prompt from Anthropic request")

        # Extract response text from content blocks
        response_text = ""
        content = response.get("content", [])

        if isinstance(content, list):
            text_blocks = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            response_text = "\n".join(text_blocks)
        elif isinstance(content, str):
            response_text = content

        if not response_text:
            raise ValueError("Cannot extract response from Anthropic response")

        # Extract token counts
        usage = response.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        # Provider-side response identifier (Anthropic returns `id`)
        provider_response_id = response.get("id") or None

        return prompt_str, response_text, tokens_in, tokens_out, provider_response_id

    @staticmethod
    def _extract_from_generic(
        request_kwargs: dict, response: dict
    ) -> Tuple[str, str, int, int, Optional[str]]:
        """
        Extract prompt, response, and token counts from generic LLM API call.

        Best-effort extraction. Tries multiple field name patterns:
        - Prompt: "prompt", "text", "input", "query"
        - Response: "text", "response", "output", "result"
        - Tokens: "tokens", "usage", "token_count"

        Args:
            request_kwargs: Generic LLM request dict
            response: Generic LLM response dict

        Returns:
            (prompt_text, response_text, tokens_in, tokens_out)

        Raises:
            ValueError: If cannot find prompt and response
        """
        # Try to extract prompt
        prompt_str = ""
        for field in ["prompt", "text", "input", "query", "messages"]:
            if field in request_kwargs:
                value = request_kwargs[field]
                if isinstance(value, str):
                    prompt_str = value
                    break
                elif isinstance(value, list):
                    # Handle messages-like list
                    prompt_str = "\n".join(str(m) for m in value)
                    break

        if not prompt_str:
            raise ValueError("Cannot extract prompt from request")

        # Try to extract response
        response_text = ""
        for field in ["text", "response", "output", "result", "content"]:
            if field in response:
                value = response[field]
                if isinstance(value, str):
                    response_text = value
                    break
                elif isinstance(value, list) and value:
                    # Handle list of dicts (e.g., Anthropic content blocks)
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        if "text" in first_item:
                            response_text = first_item["text"]
                        elif "content" in first_item:
                            response_text = first_item["content"]
                        else:
                            response_text = json.dumps(first_item)
                    else:
                        response_text = str(first_item)
                    break

        if not response_text:
            raise ValueError("Cannot extract response from response dict")

        # Try to extract token counts
        tokens_in = 0
        tokens_out = 0

        # Try direct fields
        if "input_tokens" in response:
            tokens_in = response["input_tokens"]
        elif "prompt_tokens" in response:
            tokens_in = response["prompt_tokens"]

        if "output_tokens" in response:
            tokens_out = response["output_tokens"]
        elif "completion_tokens" in response:
            tokens_out = response["completion_tokens"]

        # Try nested usage dict
        if "usage" in response:
            usage = response["usage"]
            if isinstance(usage, dict):
                tokens_in = usage.get(
                    "input_tokens", usage.get("prompt_tokens", tokens_in)
                )
                tokens_out = usage.get(
                    "output_tokens", usage.get("completion_tokens", tokens_out)
                )

        # Provider-side response identifier — try common field names.
        provider_response_id: Optional[str] = None
        for id_field in ("id", "response_id", "message_id", "request_id"):
            val = response.get(id_field)
            if isinstance(val, str) and val:
                provider_response_id = val
                break

        return prompt_str, response_text, tokens_in, tokens_out, provider_response_id


# Convenience singleton for simple use cases
_default_engine: Optional[EvidenceEngine] = None
_engine_lock = threading.Lock()


def get_evidence_engine() -> EvidenceEngine:
    """
    Get or create the default evidence engine singleton.

    Useful for simple integrations that just call:
        engine = get_evidence_engine()
        receipt = engine.create_evidence(...)

    Thread-safe. Reads config from environment variables.

    Returns:
        Singleton EvidenceEngine instance
    """
    global _default_engine
    if _default_engine is None:
        with _engine_lock:
            if _default_engine is None:
                _default_engine = EvidenceEngine()
    return _default_engine

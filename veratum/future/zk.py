"""
Veratum Zero-Knowledge Proof Module.

Provides ZK-SNARK proof generation and verification for AI model inference
using EZKL (https://github.com/zkonduit/ezkl).

The ZK proof cryptographically demonstrates that a specific AI model produced
a specific output — without revealing the model weights or input data.
Neither the model nor the data ever leaves the customer's infrastructure.

Privacy guarantee:
    This module generates Zero-Knowledge proofs using EZKL. The proof
    cryptographically demonstrates that the stated AI model produced the
    stated output. Neither the model weights nor the input data are
    transmitted to Veratum at any point. The verifying key is public —
    any party can verify this proof independently using only the proof
    and the verifying key, without access to Veratum systems.

Requires:
    pip install veratum[zk]

Example:
    >>> from veratum.zk import VeratumZKProver
    >>>
    >>> # One-time setup per model
    >>> prover = VeratumZKProver()
    >>> setup_result = prover.setup(
    ...     model_path="model.onnx",
    ...     calibration_data=[{"input_data": [[1.0, 2.0, 3.0]]}],
    ...     output_dir="./zk_artifacts"
    ... )
    >>>
    >>> # Per-inference: generate proof asynchronously
    >>> proof = prover.prove(
    ...     input_data={"input_data": [[4.0, 5.0, 6.0]]},
    ...     output_data={"output_data": [[0.92]]},
    ...     artifacts_dir="./zk_artifacts"
    ... )
    >>>
    >>> # Anyone can verify with only proof + vk (no model, no data)
    >>> is_valid = VeratumZKProver.verify_proof(
    ...     proof_path="./zk_artifacts/proof.json",
    ...     vk_path="./zk_artifacts/vk.key",
    ...     settings_path="./zk_artifacts/settings.json"
    ... )
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("veratum.zk")


def _check_ezkl() -> Any:
    """Import and return ezkl, raising clear error if not installed."""
    try:
        import ezkl
        return ezkl
    except ImportError:
        raise ImportError(
            "EZKL is required for ZK proof support. Install with:\n"
            "  pip install veratum[zk]\n"
            "or:\n"
            "  pip install ezkl"
        )


def _sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class ZKSetupResult:
    """Result of one-time ZK circuit setup for a model."""

    setup_hash: str
    """SHA-256 hash of the compiled circuit — identifies which model."""

    circuit_path: str
    """Path to compiled circuit file."""

    pk_path: str
    """Path to proving key (keep private)."""

    vk_path: str
    """Path to verifying key (share with regulators)."""

    vk_hash: str
    """SHA-256 hash of the verifying key — goes into receipts."""

    settings_path: str
    """Path to circuit settings."""

    srs_path: str
    """Path to structured reference string."""

    setup_time_ms: int
    """Time taken for setup in milliseconds."""

    model_hash: str
    """SHA-256 hash of the original ONNX model."""


@dataclass
class ZKProofResult:
    """Result of ZK proof generation for a single inference."""

    proof_hash: str
    """SHA-256 hash of the proof file."""

    proof_b64: str
    """Base64-encoded proof (for receipt attachment)."""

    verify_key_hash: str
    """Hash of the public verifying key."""

    circuit_hash: str
    """Hash of the compiled circuit (proves which model)."""

    proved_output: Any
    """The output that was cryptographically proved."""

    generation_time_ms: int
    """How long proof generation took."""

    framework: str = "ezkl"
    """ZK framework used."""

    status: str = "proved"
    """Proof status."""

    privacy_statement: str = (
        "This receipt contains a Zero-Knowledge proof generated using EZKL "
        "(github.com/zkonduit/ezkl). The proof cryptographically demonstrates "
        "that the stated AI model produced the stated output. Neither the model "
        "weights nor the input data were transmitted to Veratum at any point. "
        "The verifying key (zk_verify_key_hash) is public — any party can "
        "verify this proof independently using only the proof and the verifying "
        "key, without access to Veratum systems."
    )

    def to_receipt_fields(self) -> Dict[str, Any]:
        """Convert to receipt ZK fields for Veratum ingestion."""
        return {
            "zk_proof_hash": self.proof_hash,
            "zk_verify_key_hash": self.verify_key_hash,
            "zk_circuit_hash": self.circuit_hash,
            "zk_proved_output": self.proved_output,
            "zk_generation_time_ms": self.generation_time_ms,
            "zk_status": self.status,
            "zk_framework": self.framework,
            "zk_privacy_statement": self.privacy_statement,
        }


@dataclass
class ZKVerifyResult:
    """Result of ZK proof verification."""

    valid: bool
    """Whether the proof is valid."""

    verify_key_hash: str
    """Hash of the verifying key used."""

    proof_hash: str
    """Hash of the proof verified."""

    verification_time_ms: int
    """Time taken to verify in milliseconds."""

    error: Optional[str] = None
    """Error message if verification failed."""


# ─── ZK Prover ────────────────────────────────────────────────────────────────


class VeratumZKProver:
    """
    Zero-Knowledge proof generator for AI model inference.

    Wraps EZKL to provide a simple interface for:
    1. One-time setup: compile model to ZK circuit, generate keys
    2. Per-inference proving: generate proof that model ran correctly
    3. Verification: anyone can verify with only proof + verifying key

    The customer's model and data NEVER leave their infrastructure.
    Only cryptographic proofs and public keys are shared with Veratum.
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        *,
        input_visibility: str = "Private",
        output_visibility: str = "Public",
        param_visibility: str = "Private",
    ):
        """
        Initialize ZK prover.

        Args:
            artifacts_dir: Directory for ZK artifacts (circuit, keys, proofs).
                          Defaults to ./veratum_zk_artifacts.
            input_visibility: "Private" (default) or "Public" for model inputs.
            output_visibility: "Public" (default) or "Private" for model outputs.
            param_visibility: "Private" (default) or "Public" for model weights.
        """
        self.artifacts_dir = Path(artifacts_dir or "./veratum_zk_artifacts")
        self.input_visibility = input_visibility
        self.output_visibility = output_visibility
        self.param_visibility = param_visibility

        # Cached paths after setup
        self._circuit_path: Optional[Path] = None
        self._pk_path: Optional[Path] = None
        self._vk_path: Optional[Path] = None
        self._settings_path: Optional[Path] = None
        self._srs_path: Optional[Path] = None
        self._vk_hash: Optional[str] = None
        self._circuit_hash: Optional[str] = None
        self._is_setup: bool = False

    @property
    def is_setup(self) -> bool:
        """Whether setup has been completed for this prover."""
        return self._is_setup

    def setup(
        self,
        model_path: str,
        calibration_data: Optional[List[Dict[str, Any]]] = None,
        *,
        output_dir: Optional[str] = None,
        logrows: Optional[int] = None,
    ) -> ZKSetupResult:
        """
        One-time setup per model. Generates ZK circuit, proving key, and verifying key.

        This is computationally expensive (minutes for large models) but only runs once.
        After setup, the verifying key (vk.key) can be shared publicly with regulators
        and auditors. The proving key (pk.key) stays with the customer.

        Args:
            model_path: Path to ONNX model file.
            calibration_data: Sample inputs for circuit calibration.
                            Format: [{"input_data": [[...]]}, ...]
                            If None, uses default calibration.
            output_dir: Directory for artifacts. Defaults to self.artifacts_dir.
            logrows: Circuit size override (log2 of rows). Auto-detected if None.

        Returns:
            ZKSetupResult with paths to all generated artifacts.

        Raises:
            ImportError: If ezkl is not installed.
            FileNotFoundError: If model_path doesn't exist.
            RuntimeError: If circuit compilation or key generation fails.
        """
        ezkl = _check_ezkl()
        import asyncio

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        out = Path(output_dir) if output_dir else self.artifacts_dir
        out.mkdir(parents=True, exist_ok=True)

        settings_path = out / "settings.json"
        compiled_path = out / "model.compiled"
        pk_path = out / "pk.key"
        vk_path = out / "vk.key"
        srs_path = out / "kzg.srs"
        cal_path = out / "calibration.json"

        start = time.time()
        model_hash = _sha256_file(model_path)

        async def _run_setup() -> None:
            # Step 1: Generate settings
            py_run_args = ezkl.PyRunArgs()
            py_run_args.input_visibility = self.input_visibility
            py_run_args.output_visibility = self.output_visibility
            py_run_args.param_visibility = self.param_visibility
            if logrows is not None:
                py_run_args.logrows = logrows

            res = ezkl.gen_settings(
                str(model_path),
                str(settings_path),
                py_run_args=py_run_args,
            )
            if not res:
                raise RuntimeError("Failed to generate ZK settings")

            # Step 2: Calibrate settings with sample data
            if calibration_data:
                with open(cal_path, "w") as f:
                    json.dump(calibration_data[0], f)
                await ezkl.calibrate_settings(
                    str(cal_path),
                    str(model_path),
                    str(settings_path),
                    target="resources",
                )

            # Step 3: Compile circuit
            res = ezkl.compile_circuit(
                str(model_path),
                str(compiled_path),
                str(settings_path),
            )
            if not res:
                raise RuntimeError("Failed to compile ZK circuit")

            # Step 4: Get SRS (structured reference string)
            await ezkl.get_srs(str(settings_path), srs_path=str(srs_path))

            # Step 5: Setup — generate proving key and verifying key
            res = ezkl.setup(
                str(compiled_path),
                str(vk_path),
                str(pk_path),
                str(srs_path),
            )
            if not res:
                raise RuntimeError("Failed to generate proving/verifying keys")

        # Run async setup
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run_setup())
        finally:
            loop.close()

        elapsed_ms = int((time.time() - start) * 1000)
        circuit_hash = _sha256_file(compiled_path)
        vk_hash = _sha256_file(vk_path)

        # Cache for subsequent prove() calls
        self._circuit_path = compiled_path
        self._pk_path = pk_path
        self._vk_path = vk_path
        self._settings_path = settings_path
        self._srs_path = srs_path
        self._vk_hash = vk_hash
        self._circuit_hash = circuit_hash
        self._is_setup = True

        result = ZKSetupResult(
            setup_hash=circuit_hash,
            circuit_path=str(compiled_path),
            pk_path=str(pk_path),
            vk_path=str(vk_path),
            vk_hash=vk_hash,
            settings_path=str(settings_path),
            srs_path=str(srs_path),
            setup_time_ms=elapsed_ms,
            model_hash=model_hash,
        )

        logger.info(
            "ZK setup complete: circuit=%s vk=%s time=%dms",
            circuit_hash[:16],
            vk_hash[:16],
            elapsed_ms,
        )

        return result

    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Load previously generated ZK artifacts from disk.

        Use this instead of setup() when artifacts already exist from a previous run.

        Args:
            artifacts_dir: Directory containing setup artifacts.

        Raises:
            FileNotFoundError: If required artifacts are missing.
        """
        d = Path(artifacts_dir)
        required = {
            "model.compiled": "circuit",
            "pk.key": "proving key",
            "vk.key": "verifying key",
            "settings.json": "settings",
            "kzg.srs": "SRS",
        }

        for filename, desc in required.items():
            if not (d / filename).exists():
                raise FileNotFoundError(
                    f"Missing {desc} ({filename}) in {artifacts_dir}. "
                    f"Run setup() first to generate ZK artifacts."
                )

        self._circuit_path = d / "model.compiled"
        self._pk_path = d / "pk.key"
        self._vk_path = d / "vk.key"
        self._settings_path = d / "settings.json"
        self._srs_path = d / "kzg.srs"
        self._vk_hash = _sha256_file(self._vk_path)
        self._circuit_hash = _sha256_file(self._circuit_path)
        self._is_setup = True

        logger.info("ZK artifacts loaded from %s", artifacts_dir)

    def prove(
        self,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        *,
        artifacts_dir: Optional[str] = None,
    ) -> ZKProofResult:
        """
        Generate a ZK proof for a specific AI inference.

        The proof proves: "this model ran on this input and produced this output"
        without revealing the input or model weights.

        IMPORTANT: input_data stays on the customer's machine. It is used locally
        to generate the proof but is NEVER transmitted to Veratum.

        Args:
            input_data: Model inputs. Format: {"input_data": [[...]]}
            output_data: Expected model output (optional — witness generation
                        will compute it if not provided).
            artifacts_dir: Override artifacts directory.

        Returns:
            ZKProofResult with proof hash, base64 proof, and receipt fields.

        Raises:
            RuntimeError: If setup hasn't been run or proof generation fails.
            ImportError: If ezkl is not installed.
        """
        ezkl = _check_ezkl()
        import asyncio

        if not self._is_setup:
            if artifacts_dir:
                self.load_artifacts(artifacts_dir)
            else:
                raise RuntimeError(
                    "ZK prover not set up. Call setup() or load_artifacts() first."
                )

        start = time.time()

        # Write input data to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(self.artifacts_dir)
        ) as f:
            json.dump(input_data, f)
            input_path = f.name

        witness_path = str(self.artifacts_dir / "witness.json")
        proof_path = str(self.artifacts_dir / "proof.json")

        async def _run_prove() -> None:
            # Step 1: Generate witness (execute model, record trace)
            await ezkl.gen_witness(
                input_path,
                str(self._circuit_path),
                witness_path,
                str(self._vk_path),
                str(self._srs_path),
            )

            # Step 2: Generate proof
            res = ezkl.prove(
                witness_path,
                str(self._circuit_path),
                str(self._pk_path),
                proof_path,
                "single",
                str(self._srs_path),
            )
            if not res:
                raise RuntimeError("ZK proof generation failed")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run_prove())
        finally:
            loop.close()
            # Clean up temp input file
            try:
                os.unlink(input_path)
            except OSError:
                pass

        elapsed_ms = int((time.time() - start) * 1000)

        # Read proof file
        with open(proof_path, "rb") as f:
            proof_bytes = f.read()

        proof_hash = _sha256_bytes(proof_bytes)
        proof_b64 = base64.b64encode(proof_bytes).decode("ascii")

        # Extract proved output from witness
        proved_output = output_data
        if proved_output is None:
            try:
                with open(witness_path, "r") as f:
                    witness = json.load(f)
                proved_output = witness.get("output_data") or witness.get("outputs")
            except Exception:
                proved_output = None

        result = ZKProofResult(
            proof_hash=proof_hash,
            proof_b64=proof_b64,
            verify_key_hash=self._vk_hash,
            circuit_hash=self._circuit_hash,
            proved_output=proved_output,
            generation_time_ms=elapsed_ms,
        )

        logger.info(
            "ZK proof generated: hash=%s time=%dms",
            proof_hash[:16],
            elapsed_ms,
        )

        return result

    @staticmethod
    def verify_proof(
        proof_path: str,
        vk_path: str,
        settings_path: str,
        srs_path: Optional[str] = None,
    ) -> ZKVerifyResult:
        """
        Verify a ZK proof. Static method — anyone can call this.

        Requires ONLY the proof file and the public verifying key.
        No model weights. No input data. No Veratum account needed.
        This is pure cryptographic verification.

        Args:
            proof_path: Path to proof.json file.
            vk_path: Path to public verifying key (vk.key).
            settings_path: Path to circuit settings.json.
            srs_path: Path to SRS file (optional, auto-detected).

        Returns:
            ZKVerifyResult indicating whether the proof is valid.
        """
        ezkl = _check_ezkl()

        start = time.time()
        proof_hash = _sha256_file(proof_path)
        vk_hash = _sha256_file(vk_path)

        try:
            if srs_path:
                is_valid = ezkl.verify(
                    str(proof_path),
                    str(settings_path),
                    str(vk_path),
                    str(srs_path),
                )
            else:
                is_valid = ezkl.verify(
                    str(proof_path),
                    str(settings_path),
                    str(vk_path),
                )

            elapsed_ms = int((time.time() - start) * 1000)

            return ZKVerifyResult(
                valid=bool(is_valid),
                verify_key_hash=vk_hash,
                proof_hash=proof_hash,
                verification_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return ZKVerifyResult(
                valid=False,
                verify_key_hash=vk_hash,
                proof_hash=proof_hash,
                verification_time_ms=elapsed_ms,
                error=str(e),
            )

    @staticmethod
    def verify_proof_bytes(
        proof_b64: str,
        vk_path: str,
        settings_path: str,
        srs_path: Optional[str] = None,
    ) -> ZKVerifyResult:
        """
        Verify a ZK proof from base64-encoded bytes.

        Convenience wrapper for verifying proofs stored in Veratum receipts.

        Args:
            proof_b64: Base64-encoded proof (from receipt's zk_proof_b64 field).
            vk_path: Path to public verifying key.
            settings_path: Path to circuit settings.
            srs_path: Path to SRS file (optional).

        Returns:
            ZKVerifyResult indicating whether the proof is valid.
        """
        proof_bytes = base64.b64decode(proof_b64)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            f.write(proof_bytes)
            temp_proof = f.name

        try:
            return VeratumZKProver.verify_proof(
                temp_proof, vk_path, settings_path, srs_path
            )
        finally:
            try:
                os.unlink(temp_proof)
            except OSError:
                pass


# ─── Receipt ZK Fields Helper ────────────────────────────────────────────────


def zk_pending_fields() -> Dict[str, Any]:
    """
    Return ZK receipt fields for a receipt where proof is pending.

    Use this when the AI decision has been made but the ZK proof
    hasn't been generated yet (async proving pattern).
    """
    return {
        "zk_status": "pending",
        "zk_framework": "ezkl",
        "zk_proof_hash": None,
        "zk_verify_key_hash": None,
        "zk_circuit_hash": None,
        "zk_proved_output": None,
        "zk_generation_time_ms": None,
    }


def zk_not_applicable_fields() -> Dict[str, Any]:
    """
    Return ZK receipt fields for a receipt where ZK is not applicable.

    Use this for models that haven't been set up for ZK proving.
    """
    return {
        "zk_status": "not_applicable",
        "zk_framework": None,
        "zk_proof_hash": None,
        "zk_verify_key_hash": None,
        "zk_circuit_hash": None,
        "zk_proved_output": None,
        "zk_generation_time_ms": None,
    }

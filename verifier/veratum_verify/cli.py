"""
Veratum Verification CLI — Command-line interface for receipt verification.

Usage:
  veratum-verify receipt <receipt.json>
  veratum-verify chain <receipts.json>
  veratum-verify inclusion --leaf HASH --proof proof.json --root ROOT --size N
  veratum-verify consistency --old-size N --new-size M --old-root OLD --new-root NEW --proof proof.json
"""

import argparse
import json
import sys
from typing import Any, Dict

from veratum_verify.core import (
    ReceiptVerifier,
    verify_inclusion,
    verify_consistency,
)

# Windows console (CP1252) cannot encode Unicode check/cross marks.
# Reconfigure stdout/stderr to UTF-8 so the Unicode symbols render correctly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, Exception):
        # Python < 3.7 or non-reconfigurable stream — fall back silently
        pass


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_check(name: str, passed: bool, details: str = ""):
    """Print a single check result with colored icon."""
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    if details:
        print(f"{icon} {name:30s} [{status}] {details}")
    else:
        print(f"{icon} {name:30s} [{status}]")


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print("-" * 70)


def load_json_file(filepath: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: File not found: {filepath}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON in {filepath}: {e}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)


def cmd_receipt(args: argparse.Namespace) -> int:
    """Verify a single receipt."""
    receipt = load_json_file(args.receipt)

    print_header("Veratum Receipt Verification")
    print(f"Receipt ID: {receipt.get('receipt_id', 'N/A')}")

    verifier = ReceiptVerifier()
    result = verifier.verify_receipt(receipt)

    print()
    for check_name, passed in result.checks.items():
        print_check(check_name, passed)

    if result.errors:
        print()
        print(f"{Colors.RED}Errors:{Colors.RESET}")
        for error in result.errors:
            print(f"  • {error}")

    print()
    if result.valid:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Receipt is valid{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Receipt is invalid{Colors.RESET}")
        return 1


def cmd_chain(args: argparse.Namespace) -> int:
    """Verify a chain of receipts."""
    receipts_data = load_json_file(args.chain)

    # Handle both single receipt and list of receipts
    if isinstance(receipts_data, dict):
        receipts = [receipts_data]
    elif isinstance(receipts_data, list):
        receipts = receipts_data
    else:
        print(f"{Colors.RED}Error: Expected dict or list of receipts{Colors.RESET}", file=sys.stderr)
        return 1

    print_header("Veratum Chain Verification")
    print(f"Chain length: {len(receipts)} receipt(s)")

    verifier = ReceiptVerifier()
    result = verifier.verify_chain(receipts)

    print()
    print_check("Chain integrity", result.valid)

    if result.breaks:
        print()
        print(f"{Colors.RED}Chain breaks at indices:{Colors.RESET}")
        for break_idx in result.breaks:
            print(f"  • Index {break_idx}")

    if result.errors:
        print()
        print(f"{Colors.RED}Errors:{Colors.RESET}")
        for error in result.errors:
            print(f"  • {error}")

    print()
    if result.valid:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Chain is valid{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Chain is invalid{Colors.RESET}")
        return 1


def cmd_inclusion(args: argparse.Namespace) -> int:
    """Verify an inclusion proof."""
    print_header("Veratum Inclusion Proof Verification")
    print(f"Leaf hash: {args.leaf[:16]}...")
    print(f"Tree size: {args.size}")
    print(f"Leaf index: {args.leaf_index if args.leaf_index is not None else 'N/A'}")

    # Load proof from file if provided
    if args.proof:
        proof_data = load_json_file(args.proof)
        if isinstance(proof_data, list):
            proof = proof_data
        elif isinstance(proof_data, dict) and "proof" in proof_data:
            proof = proof_data["proof"]
        else:
            proof = proof_data
    else:
        proof = []

    # If leaf_index not provided, try to extract from proof file
    leaf_index = args.leaf_index
    if leaf_index is None and args.proof:
        proof_data = load_json_file(args.proof)
        if isinstance(proof_data, dict) and "leaf_index" in proof_data:
            leaf_index = proof_data["leaf_index"]

    if leaf_index is None:
        print(f"{Colors.RED}Error: --leaf-index required (or provide in proof file){Colors.RESET}",
              file=sys.stderr)
        return 1

    print()
    valid = verify_inclusion(args.leaf, proof, args.size, args.root, leaf_index)
    print_check("Proof verification", valid)

    print()
    if valid:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Inclusion proof is valid{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Inclusion proof is invalid{Colors.RESET}")
        return 1


def cmd_consistency(args: argparse.Namespace) -> int:
    """Verify a consistency proof."""
    print_header("Veratum Consistency Proof Verification")
    print(f"Old tree size: {args.old_size}")
    print(f"New tree size: {args.new_size}")
    print(f"Old root: {args.old_root[:16]}...")
    print(f"New root: {args.new_root[:16]}...")

    # Load proof from file if provided
    if args.proof:
        proof_data = load_json_file(args.proof)
        if isinstance(proof_data, list):
            proof = proof_data
        elif isinstance(proof_data, dict) and "proof" in proof_data:
            proof = proof_data["proof"]
        else:
            proof = proof_data
    else:
        proof = []

    print()
    valid = verify_consistency(args.old_size, args.new_size, args.old_root, args.new_root, proof)
    print_check("Proof verification", valid)

    print()
    if valid:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Consistency proof is valid{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Consistency proof is invalid{Colors.RESET}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Veratum Verification Toolkit - Independently verify Veratum compliance receipts"
    )

    subparsers = parser.add_subparsers(dest="command", help="Verification command")

    # receipt subcommand
    receipt_parser = subparsers.add_parser("receipt", help="Verify a single receipt")
    receipt_parser.add_argument("receipt", help="Path to receipt JSON file")
    receipt_parser.set_defaults(func=cmd_receipt)

    # chain subcommand
    chain_parser = subparsers.add_parser("chain", help="Verify a chain of receipts")
    chain_parser.add_argument("chain", help="Path to receipts JSON file")
    chain_parser.set_defaults(func=cmd_chain)

    # inclusion subcommand
    inclusion_parser = subparsers.add_parser("inclusion", help="Verify an inclusion proof")
    inclusion_parser.add_argument("--leaf", required=True, help="Leaf hash (hex string)")
    inclusion_parser.add_argument("--proof", help="Path to proof JSON file")
    inclusion_parser.add_argument("--root", required=True, help="Root hash (hex string)")
    inclusion_parser.add_argument("--size", type=int, required=True, help="Tree size")
    inclusion_parser.add_argument("--leaf-index", type=int, help="Leaf index (0-based)")
    inclusion_parser.set_defaults(func=cmd_inclusion)

    # consistency subcommand
    consistency_parser = subparsers.add_parser("consistency", help="Verify a consistency proof")
    consistency_parser.add_argument("--old-size", type=int, required=True, help="Old tree size")
    consistency_parser.add_argument("--new-size", type=int, required=True, help="New tree size")
    consistency_parser.add_argument("--old-root", required=True, help="Old root hash (hex string)")
    consistency_parser.add_argument("--new-root", required=True, help="New root hash (hex string)")
    consistency_parser.add_argument("--proof", help="Path to proof JSON file")
    consistency_parser.set_defaults(func=cmd_consistency)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Convert WPT WebNN Tests from JavaScript to JSON

This script converts W3C Web Platform Tests for WebNN from their JavaScript
format to JSON format suitable for testing the rustnn implementation.

Usage:
    python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --operation reduce_sum
    python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --all-operations
    python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --list-operations

The script handles:
- Parsing JavaScript test files
- Converting test data structures to JSON
- Preserving tolerance specifications
- Recording source metadata for future updates
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys


# Map of WebNN operation names to WPT test file names
OPERATION_TO_WPT_FILE = {
    # Binary operations
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "matmul": "matmul",
    "pow": "pow",
    "min": "min",
    "max": "max",

    # Activation functions
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "softmax": "softmax",
    "elu": "elu",
    "leaky_relu": "leaky_relu",
    "linear": "linear",
    "hard_sigmoid": "hard_sigmoid",
    "hard_swish": "hard_swish",

    # Reduction operations
    "reduce_sum": "reduce",
    "reduce_mean": "reduce",
    "reduce_max": "reduce",
    "reduce_min": "reduce",
    "reduce_product": "reduce",
    "reduce_l1": "reduce",
    "reduce_l2": "reduce",
    "reduce_log_sum": "reduce",
    "reduce_log_sum_exp": "reduce",
    "reduce_sum_square": "reduce",

    # Pooling operations
    "average_pool2d": "pool2d",
    "max_pool2d": "pool2d",
    "global_average_pool": "global_average_pool",
    "global_max_pool": "global_max_pool",

    # Convolution
    "conv2d": "conv2d",
    "conv_transpose2d": "conv_transpose2d",

    # Normalization
    "batch_normalization": "batch_normalization",
    "instance_normalization": "instance_normalization",
    "layer_normalization": "layer_normalization",

    # Shape operations
    "reshape": "reshape",
    "transpose": "transpose",
    "concat": "concat",
    "split": "split",
    "slice": "slice",
    "expand": "expand",
    "gather": "gather",

    # Element-wise operations
    "abs": "abs",
    "ceil": "ceil",
    "floor": "floor",
    "exp": "exp",
    "log": "log",
    "neg": "neg",
    "sqrt": "sqrt",
    "cast": "cast",
    "clamp": "clamp",

    # Logical operations
    "equal": "equal",
    "greater": "greater",
    "greater_or_equal": "greater_or_equal",
    "lesser": "lesser",
    "lesser_or_equal": "lesser_or_equal",
    "logical_not": "logical_not",
}


def get_git_commit(repo_path: Path) -> str:
    """Get the current git commit SHA from a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def find_wpt_test_files(wpt_repo: Path, operation: str) -> List[Path]:
    """
    Find WPT test files for a given operation.

    Args:
        wpt_repo: Path to WPT repository
        operation: Operation name (e.g., "reduce_sum")

    Returns:
        List of paths to test files
    """
    wpt_test_file = OPERATION_TO_WPT_FILE.get(operation)
    if not wpt_test_file:
        return []

    test_dirs = [
        wpt_repo / "webnn" / "conformance_tests",
        wpt_repo / "webnn" / "validation_tests",
    ]

    test_files = []
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        # Look for test files matching the operation
        # Pattern: <operation>.https.any.js or <operation>_*.https.any.js
        patterns = [
            f"{wpt_test_file}.https.any.js",
            f"{wpt_test_file}_*.https.any.js",
        ]

        for pattern in patterns:
            test_files.extend(test_dir.glob(pattern))

    return sorted(test_files)


def parse_js_array_simple(js_content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Simple parser for JavaScript test arrays.

    This is a basic parser that extracts test case objects from JavaScript
    arrays. It uses regex patterns to find test cases and JSON-like structures.

    Note: This is not a full JavaScript parser. It works for the structured
    format used in WPT tests but may not handle all JavaScript syntax.

    Args:
        js_content: JavaScript file content

    Returns:
        List of test case dictionaries, or None if parsing fails
    """
    # Try to find test array declarations
    # Pattern: const xxxTests = [ ... ];
    array_pattern = r'const\s+(\w+Tests)\s*=\s*(\[[\s\S]*?\]);'

    matches = re.findall(array_pattern, js_content, re.MULTILINE)

    if not matches:
        return None

    # For now, return empty list to indicate we found test structure
    # Full parsing would require a JavaScript AST parser
    # This is a placeholder for the actual implementation
    print(f"  Found {len(matches)} test arrays")
    print(f"  Note: Full JavaScript parsing requires manual conversion or JS parser")
    return []


def convert_operation_tests(
    wpt_repo: Path,
    operation: str,
    output_dir: Path,
    category: str = "conformance"
) -> Dict[str, Any]:
    """
    Convert WPT tests for a single operation to JSON format.

    Args:
        wpt_repo: Path to WPT repository
        operation: Operation name
        output_dir: Output directory for JSON files
        category: Test category ("conformance" or "validation")

    Returns:
        Conversion result with metadata
    """
    test_files = find_wpt_test_files(wpt_repo, operation)

    if not test_files:
        return {
            "operation": operation,
            "status": "not_found",
            "message": f"No WPT test files found for {operation}"
        }

    # Filter by category
    category_files = [
        f for f in test_files
        if category in str(f.parent.name)
    ]

    if not category_files:
        return {
            "operation": operation,
            "status": "no_category_files",
            "message": f"No {category} test files found"
        }

    # Read the first matching file
    test_file = category_files[0]
    with open(test_file, 'r', encoding='utf-8') as f:
        js_content = f.read()

    # Get git metadata
    wpt_commit = get_git_commit(wpt_repo)
    wpt_version = datetime.now().strftime("%Y-%m-%d")

    # Parse test cases
    test_cases = parse_js_array_simple(js_content)

    # Create output structure
    output_data = {
        "operation": operation,
        "wpt_version": wpt_version,
        "wpt_commit": wpt_commit,
        "source_file": str(test_file.relative_to(wpt_repo)),
        "conversion_note": (
            "This file was auto-generated from WPT JavaScript tests. "
            "Manual verification recommended. "
            "Some test cases may require manual conversion from JavaScript syntax."
        ),
        "tests": test_cases if test_cases else []
    }

    # Write output file
    output_file = output_dir / f"{operation}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    return {
        "operation": operation,
        "status": "converted",
        "output_file": str(output_file),
        "source_file": str(test_file),
        "test_count": len(test_cases) if test_cases else 0,
        "requires_manual_review": not test_cases or len(test_cases) == 0
    }


def list_available_operations(wpt_repo: Path) -> List[str]:
    """List all operations that have WPT test files."""
    available = []
    for operation in sorted(OPERATION_TO_WPT_FILE.keys()):
        test_files = find_wpt_test_files(wpt_repo, operation)
        if test_files:
            available.append(operation)
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Convert WPT WebNN tests from JavaScript to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single operation
  python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --operation reduce_sum

  # Convert multiple operations
  python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --operations reduce_sum,relu,add

  # List available operations
  python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --list-operations

  # Convert all implemented operations
  python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --all-operations

Note:
  This script provides a framework for converting WPT tests. Due to the complexity
  of parsing JavaScript, some test files may require manual conversion. The script
  will create JSON template files that can be manually populated.
        """
    )

    parser.add_argument(
        "--wpt-repo",
        type=Path,
        required=True,
        help="Path to WPT repository (https://github.com/web-platform-tests/wpt)"
    )

    parser.add_argument(
        "--operation",
        type=str,
        help="Single operation to convert (e.g., reduce_sum)"
    )

    parser.add_argument(
        "--operations",
        type=str,
        help="Comma-separated list of operations to convert"
    )

    parser.add_argument(
        "--all-operations",
        action="store_true",
        help="Convert all available operations"
    )

    parser.add_argument(
        "--list-operations",
        action="store_true",
        help="List all operations with WPT tests"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/wpt_data"),
        help="Output directory for JSON files (default: tests/wpt_data)"
    )

    parser.add_argument(
        "--category",
        choices=["conformance", "validation", "both"],
        default="both",
        help="Test category to convert (default: both)"
    )

    args = parser.parse_args()

    # Validate WPT repo path
    if not args.wpt_repo.exists():
        print(f"Error: WPT repository not found at {args.wpt_repo}", file=sys.stderr)
        print("Clone it with: git clone https://github.com/web-platform-tests/wpt.git", file=sys.stderr)
        return 1

    webnn_dir = args.wpt_repo / "webnn"
    if not webnn_dir.exists():
        print(f"Error: WebNN tests not found in {webnn_dir}", file=sys.stderr)
        return 1

    # List operations mode
    if args.list_operations:
        operations = list_available_operations(args.wpt_repo)
        print("Available operations with WPT tests:")
        for op in operations:
            test_files = find_wpt_test_files(args.wpt_repo, op)
            print(f"  - {op:30s} ({len(test_files)} test files)")
        return 0

    # Determine operations to convert
    operations = []
    if args.operation:
        operations = [args.operation]
    elif args.operations:
        operations = [op.strip() for op in args.operations.split(",")]
    elif args.all_operations:
        operations = list_available_operations(args.wpt_repo)
    else:
        print("Error: Must specify --operation, --operations, --all-operations, or --list-operations", file=sys.stderr)
        return 1

    # Convert operations
    categories = ["conformance", "validation"] if args.category == "both" else [args.category]

    results = []
    for operation in operations:
        print(f"\nConverting {operation}...")
        for category in categories:
            print(f"  Category: {category}")
            output_dir = args.output / category
            result = convert_operation_tests(args.wpt_repo, operation, output_dir, category)
            results.append(result)

            if result["status"] == "converted":
                status_icon = "⚠️" if result.get("requires_manual_review") else "✅"
                print(f"  {status_icon} Output: {result['output_file']}")
                if result.get("requires_manual_review"):
                    print(f"     Note: Requires manual test case population")
            else:
                print(f"  ❌ {result['message']}")

    # Summary
    print("\n" + "="*70)
    print("Conversion Summary:")
    converted = [r for r in results if r["status"] == "converted"]
    needs_review = [r for r in converted if r.get("requires_manual_review")]

    print(f"  Total operations processed: {len(operations)}")
    print(f"  Successfully converted: {len(converted)}")
    print(f"  Requires manual review: {len(needs_review)}")

    if needs_review:
        print("\nOperations requiring manual review:")
        for result in needs_review:
            print(f"  - {result['operation']}")
        print("\nNext steps:")
        print("  1. Review the generated JSON template files")
        print("  2. Manually add test cases from the JavaScript source files")
        print("  3. Refer to: docs/wpt-integration-plan.md for guidance")

    return 0


if __name__ == "__main__":
    sys.exit(main())

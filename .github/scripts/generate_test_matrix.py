#!/usr/bin/env python3
"""
Generate dynamic test matrices for GitHub Actions workflows.

This script reads the model registry and generates JSON matrices for different
test suites. It supports manual exclusions and an allowlist filter passed as
command-line arguments.

Usage:
    python generate_test_matrix.py [--exclude MODEL_PATH ...] [--only MODEL_PATH ...]

Example:
    python generate_test_matrix.py --exclude granite-vision phi4
    python generate_test_matrix.py --only Qwen/Qwen3-0.6B ministral/Ministral-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from tests/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test_matrix_config import MATRIX_CONFIG, MatrixKey  # noqa: E402


def generate_matrices(exclude_models=None, only_models=None):
    """
    Generate test matrices from the model registry.

    Args:
        exclude_models: List of model paths to exclude from all matrices
        only_models: If non-empty, restrict all matrices to just these model
            paths (applied after exclusions). Empty/None = no restriction.

    Returns:
        dict: Dictionary containing the per-suite and combined matrix lists
    """
    exclude_models = set(exclude_models or [])
    only_models = set(only_models or [])

    # Every category has a smallest-per-adapter representative list (default)
    # and an unreduced "every registered path" list (ALL_*_PATHS, used by
    # --only so a caller can target a non-representative checkpoint, e.g. a
    # larger model sharing an adapter with a smaller default). Adding a new
    # category later just means adding a row here.
    paths = {}
    for name, (_, representative_paths, all_paths) in MATRIX_CONFIG.items():
        source = all_paths if only_models else representative_paths
        selected = [p for p in source if p not in exclude_models]
        if only_models:
            selected = [p for p in selected if p in only_models]
        paths[name] = selected

    # Feeds test_load_spyre.py's five model_path-parametrized suites.
    paths[MatrixKey.COMBINED] = (
        paths[MatrixKey.CAUSAL]
        + paths[MatrixKey.EMBED]
        + paths[MatrixKey.MASKED_LM]
        + paths[MatrixKey.QUESTION_ANSWERING]
        + paths[MatrixKey.TOKEN_CLASSIFICATION]
    )
    return paths


def format_for_github_actions(matrices):
    """
    Format matrices as GitHub Actions JSON output.

    Args:
        matrices: Dictionary with matrix lists

    Returns:
        dict: Dictionary with JSON-stringified matrices
    """
    return {key.value: json.dumps(matrices[key]) for key in MatrixKey}


def write_github_output(outputs):
    """
    Write outputs to GitHub Actions output file.

    Args:
        outputs: Dictionary of output_name -> output_value
    """
    import os

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        # Not running in GitHub Actions, print to stdout for debugging
        print("Not running in GitHub Actions. Output would be:")
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    with open(github_output, "a") as f:
        for key, value in outputs.items():
            # GitHub Actions multiline output format
            f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic test matrices for GitHub Actions"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Model keys to exclude from all matrices (e.g., granite-vision phi4)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="If given, restrict all matrices to just these model paths "
        "(e.g., Qwen/Qwen3-0.6B ministral/Ministral-3B-Instruct)",
    )
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--publish-file", type=Path)

    args = parser.parse_args()

    if args.publish_file:
        stored = json.loads(args.publish_file.read_text())
        missing = {key.value for key in MatrixKey} - set(stored)
        if missing:
            parser.error(f"matrix file is missing keys: {', '.join(sorted(missing))}")
        write_github_output(
            {key.value: json.dumps(stored[key.value]) for key in MatrixKey}
        )
        return

    # Generate matrices
    matrices = generate_matrices(exclude_models=args.exclude, only_models=args.only)

    # Print summary for workflow logs
    print("Generated test matrices:")
    for key in MatrixKey:
        print(f"  {key.value} ({len(matrices[key])}): {', '.join(matrices[key])}")

    if args.exclude:
        print(f"\nExcluded models: {', '.join(args.exclude)}")
    if args.only:
        print(f"\nRestricted to models: {', '.join(args.only)}")

    # Format for GitHub Actions
    outputs = format_for_github_actions(matrices)
    if args.output_file:
        args.output_file.write_text(
            json.dumps({key.value: matrices[key] for key in MatrixKey})
        )

    # Write to GitHub Actions output
    write_github_output(outputs)

    print("\nMatrices written to GitHub Actions output.")


if __name__ == "__main__":
    main()

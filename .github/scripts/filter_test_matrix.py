#!/usr/bin/env python3
"""Filter generated CI model matrices using the changed-file import impact.

This is deliberately a second pass: ``generate_test_matrix.py`` remains the
source of the complete matrices, and disabling this filter restores them
without changing model-selection behavior.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_matrix_config import MATRIX_CONFIG, MatrixKey  # noqa: E402

DOC_PREFIXES = ("docs/",)
ROOT_DOC_PATHS = {"README.md", "ARCHITECTURE.md", "ONBOARDING.md", "CLAUDE.md"}


def _is_documentation_path(path: str) -> bool:
    return path.startswith(DOC_PREFIXES) or path in ROOT_DOC_PATHS


def _is_hf_adapter_path(path: str) -> bool:
    candidate = Path(path)
    return (
        candidate.suffix == ".py"
        and len(candidate.parts) > 1
        and candidate.parts[0] == "hf_adapters"
        and candidate.name.startswith("hf_")
    )


def _module_for_path(path: str) -> str | None:
    candidate = Path(path)
    if (
        candidate.suffix != ".py"
        or not candidate.parts
        or candidate.parts[0] != "hf_adapters"
    ):
        return None
    parts = list(candidate.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_imports(path: Path, module: str) -> set[str]:
    """Return imported ``hf_adapters`` modules, without importing the code."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("hf_adapters.")
            )
        elif isinstance(node, ast.ImportFrom):
            imported_module = node.module or ""
            if node.level:
                package = module.split(".")[:-1]
                keep = len(package) - (node.level - 1)
                imported_module = ".".join(
                    [*package[: max(keep, 0)], *imported_module.split(".")]
                ).rstrip(".")
            if imported_module == "hf_adapters":
                imports.update(f"hf_adapters.{alias.name}" for alias in node.names)
            elif imported_module.startswith("hf_adapters."):
                imports.add(imported_module)
    return imports


def dependency_graph(root: Path = ROOT) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in (root / "hf_adapters").rglob("*.py"):
        module = _module_for_path(path.relative_to(root).as_posix())
        if module:
            graph[module] = _local_imports(path, module)
    return graph


def _depends_on(module: str, changed: set[str], graph: dict[str, set[str]]) -> bool:
    pending = [module]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in changed:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return False


def affected_adapters(changed_files: list[str], root: Path = ROOT) -> set[str] | None:
    """Return adapter filenames, or ``None`` when all models must be retained."""
    normalized = [path.removeprefix("./") for path in changed_files if path.strip()]
    if not normalized:
        return None

    # Limit tests only for two deliberately narrow PR shapes. In particular,
    # mixing documentation with adapter changes retains the complete matrices.
    if all(_is_documentation_path(path) for path in normalized):
        return set()
    if not all(_is_hf_adapter_path(path) for path in normalized):
        return None

    changed_modules = {
        module for path in normalized if (module := _module_for_path(path)) is not None
    }

    graph = dependency_graph(root)
    adapter_names = {
        info["adapter"]
        for models, _, _ in MATRIX_CONFIG.values()
        for info in models.values()
    }
    affected = {
        adapter
        for adapter in adapter_names
        if _depends_on(f"hf_adapters.{Path(adapter).stem}", changed_modules, graph)
    }
    # A changed production module with no known consumers is ambiguous. Keep all.
    return affected or None


def filter_matrices(
    matrices: dict[str, list[str]], adapters: set[str] | None
) -> dict[str, list[str]]:
    if adapters is None:
        return matrices
    paths_by_adapter: dict[str, set[str]] = {}
    for models, _, _ in MATRIX_CONFIG.values():
        for info in models.values():
            paths_by_adapter.setdefault(info["adapter"], set()).add(info["path"])
    keep = (
        set().union(*(paths_by_adapter.get(adapter, set()) for adapter in adapters))
        if adapters
        else set()
    )
    filtered = {
        key.value: [path for path in matrices[key.value] if path in keep]
        for key in MatrixKey
        if key is not MatrixKey.COMBINED
    }
    filtered[MatrixKey.COMBINED.value] = (
        filtered[MatrixKey.CAUSAL.value]
        + filtered[MatrixKey.EMBED.value]
        + filtered[MatrixKey.MASKED_LM.value]
        + filtered[MatrixKey.QUESTION_ANSWERING.value]
        + filtered[MatrixKey.TOKEN_CLASSIFICATION.value]
    )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changed-files", required=True, help="newline-delimited changed-files file"
    )
    parser.add_argument("--matrix-file", required=True, type=Path)
    args = parser.parse_args()

    # obtain the list of changed files
    changed_files = Path(args.changed_files).read_text().splitlines()
    print("Changed files:", ", ".join(changed_files) or "(none)")

    # figure out the list of affected adapters (directly or indirectly)
    adapters = affected_adapters(changed_files)
    print(
        "Affected adapters:",
        (
            "ALL (conservative fallback)"
            if adapters is None
            else ", ".join(sorted(adapters)) or "none"
        ),
    )

    matrices = json.loads(args.matrix_file.read_text())
    filtered = filter_matrices(matrices, adapters)
    print("Filtered matrices size:")
    for key, paths in filtered.items():
        print(f"  {key}: {len(paths)} model(s)")

    # Atomic replacement leaves the complete matrix file untouched if filtering
    # fails before a valid filtered document has been fully written.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.matrix_file.parent, delete=False
    ) as stream:
        json.dump(filtered, stream)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, args.matrix_file)


if __name__ == "__main__":
    main()

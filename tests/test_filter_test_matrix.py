import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / ".github/scripts/filter_test_matrix.py"
GENERATOR = Path(__file__).parents[1] / ".github/scripts/generate_test_matrix.py"
SPEC = importlib.util.spec_from_file_location("filter_test_matrix", SCRIPT)
filter_test_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(filter_test_matrix)
test_matrix_config = importlib.import_module("test_matrix_config")


def test_direct_adapter_change_selects_that_adapter():
    assert filter_test_matrix.affected_adapters(["hf_adapters/hf_llama.py"]) == {
        "hf_llama.py"
    }


def test_common_change_selects_all_dependent_registered_adapters():
    affected = filter_test_matrix.affected_adapters(["hf_adapters/hf_common.py"])
    assert affected is not None
    assert "hf_llama.py" in affected
    assert "hf_qwen3.py" in affected


def test_transitive_dependency_is_followed():
    affected = filter_test_matrix.affected_adapters(["hf_adapters/hf_mistral.py"])
    assert affected is not None
    assert "hf_mistral.py" in affected
    assert "hf_mistral3.py" in affected


def test_dependency_graph_resolves_relative_imports(tmp_path):
    package = tmp_path / "hf_adapters"
    package.mkdir()
    (package / "hf_example.py").write_text("from .hf_helper import value\n")
    (package / "hf_helper.py").write_text("value = 1\n")

    graph = filter_test_matrix.dependency_graph(tmp_path)

    assert graph["hf_adapters.hf_example"] == {"hf_adapters.hf_helper"}


def test_docs_only_change_selects_no_adapters():
    assert (
        filter_test_matrix.affected_adapters(["README.md", "docs/design.md"]) == set()
    )


def test_docs_mixed_with_adapter_change_keeps_everything():
    assert (
        filter_test_matrix.affected_adapters(
            ["hf_adapters/hf_llama.py", "docs/design.md"]
        )
        is None
    )


def test_non_hf_prefixed_file_in_adapter_directory_keeps_everything():
    assert filter_test_matrix.affected_adapters(["hf_adapters/st_backend.py"]) is None
    assert (
        filter_test_matrix.affected_adapters(["hf_adapters/auto_spyre_model.py"])
        is None
    )


def test_empty_changed_file_list_keeps_everything():
    assert filter_test_matrix.affected_adapters([]) is None


def test_unknown_or_ci_change_keeps_everything():
    assert filter_test_matrix.affected_adapters(["setup.cfg"]) is None
    assert filter_test_matrix.affected_adapters(["requirements.txt"]) is None
    assert (
        filter_test_matrix.affected_adapters(
            [".github/workflows/test_pull_request.yaml"]
        )
        is None
    )


def test_filter_preserves_generated_order_and_rebuilds_combined():
    matrices = {
        "causal_matrix": ["gpt2", "Qwen/Qwen3-0.6B"],
        "embed_matrix": ["Qwen/Qwen3-Embedding-0.6B"],
        "vision_matrix": [],
        "masked_lm_matrix": [],
        "question_answering_matrix": [],
        "token_classification_matrix": [],
        "reranker_matrix": [],
        "combined_matrix": ["stale"],
    }
    result = filter_test_matrix.filter_matrices(matrices, {"hf_qwen3.py"})
    assert result["causal_matrix"] == ["Qwen/Qwen3-0.6B"]
    assert result["embed_matrix"] == ["Qwen/Qwen3-Embedding-0.6B"]
    assert result["combined_matrix"] == result["causal_matrix"] + result["embed_matrix"]


def test_matrix_config_covers_every_non_derived_key():
    assert set(filter_test_matrix.MATRIX_CONFIG) == set(
        filter_test_matrix.MatrixKey
    ) - {filter_test_matrix.MatrixKey.COMBINED}


def test_matrix_config_covers_every_model_registry():
    test_matrix_config._validate_matrix_config()


def test_main_filters_the_durable_matrix_file(monkeypatch, tmp_path):
    matrices = {
        key.value: ["gpt2"] if key is filter_test_matrix.MatrixKey.CAUSAL else []
        for key in filter_test_matrix.MatrixKey
    }
    matrices[filter_test_matrix.MatrixKey.COMBINED.value] = ["gpt2"]
    matrix_file = tmp_path / "matrices.json"
    matrix_file.write_text(json.dumps(matrices))
    changed_files = tmp_path / "changed-files.txt"
    changed_files.write_text("README.md\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_test_matrix.py",
            "--changed-files",
            str(changed_files),
            "--matrix-file",
            str(matrix_file),
        ],
    )

    filter_test_matrix.main()

    assert json.loads(matrix_file.read_text()) == {
        key.value: [] for key in filter_test_matrix.MatrixKey
    }


def test_failed_filter_leaves_durable_matrix_file_unchanged(monkeypatch, tmp_path):
    matrix_file = tmp_path / "matrices.json"
    original = '{"causal_matrix": ["gpt2"]}'
    matrix_file.write_text(original)
    changed_files = tmp_path / "changed-files.txt"
    changed_files.write_text("README.md\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_test_matrix.py",
            "--changed-files",
            str(changed_files),
            "--matrix-file",
            str(matrix_file),
        ],
    )

    try:
        filter_test_matrix.main()
    except KeyError:
        pass
    else:
        raise AssertionError("incomplete matrix should fail filtering")

    assert matrix_file.read_text() == original


def test_generate_file_can_be_published_as_github_outputs(tmp_path):
    matrix_file = tmp_path / "matrices.json"
    github_output = tmp_path / "github-output.txt"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--only",
            "Qwen/Qwen3-0.6B",
            "--output-file",
            str(matrix_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(github_output)
    subprocess.run(
        [sys.executable, str(GENERATOR), "--publish-file", str(matrix_file)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    published = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert json.loads(published["causal_matrix"]) == ["Qwen/Qwen3-0.6B"]
    assert set(published) == {key.value for key in filter_test_matrix.MatrixKey}

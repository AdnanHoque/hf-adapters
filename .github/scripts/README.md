# GitHub Actions Scripts

## generate_test_matrix.py

Dynamically generates test matrices for GitHub Actions workflows from the model registry.

### Purpose

This script ensures that CI test matrices stay synchronized with the model registry (`tests/model_registry.py`) and adapter mappings (`hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`). When new adapters are added, the CI automatically tests them without manual workflow updates.

### How It Works

1. Imports the representative path lists from `tests/model_registry.py`.
2. Generates one JSON array per task suite, including `causal_matrix`,
   `embed_matrix`, `masked_lm_matrix`, `vision_matrix`, and `reranker_matrix`.
   `combined_matrix` remains the causal + embedding matrix used by shared jobs.
3. Outputs matrices in GitHub Actions format for consumption by test jobs.

### Usage

```bash
# Generate all matrices
python .github/scripts/generate_test_matrix.py

# Exclude specific models (e.g., temporarily broken models)
python .github/scripts/generate_test_matrix.py --exclude granite-vision phi4

# In GitHub Actions workflow
- name: Generate matrices
  id: generate
  run: |
    python .github/scripts/generate_test_matrix.py --exclude granite-vision
```

### Excluding Models

To temporarily exclude models from all test matrices:

1. Edit `.github/workflows/test_pull_request.yaml`
2. Find the `generate-matrix` job
3. Add model keys to the `--exclude` list:
   ```yaml
   python .github/scripts/generate_test_matrix.py --exclude granite-vision phi4 other-model
   ```

### Adding New Models

To add a new model to CI:

1. Add the model to the appropriate task registry in `tests/model_registry.py`.
2. Ensure the model's adapter is in `hf_adapters/auto_spyre_model.py::CONFIG_TO_ADAPTER_MODULE_MAPPING`
3. The CI will automatically include it in the next run

No workflow changes needed!

### Dependencies

- Python 3.11+
- transformers
- torch

### Optional PR impact filtering

The reusable test workflow first writes the complete generated matrices to one
JSON file. When `filter_models_by_changes` is enabled for a pull request,
`filter_test_matrix.py` atomically replaces that file with matrices restricted
to adapters affected by the changed Python modules. This narrowing is attempted
only when every changed file is an `hf_adapters/hf_*.py` file, or when every
change is documentation under `docs/` or one of the recognized root docs.
Mixed or unrecognized changes retain the complete matrices. A final publisher
exposes the file through the usual individual GitHub Actions outputs. If
filtering is skipped or fails, the original complete file remains available.

`test_matrix_config.py` is the shared source of truth for matrix keys, model
registries, representative paths, and complete path lists.

These are installed in the `generate-matrix` job before running the script.

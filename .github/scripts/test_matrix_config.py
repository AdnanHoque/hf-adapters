"""Single source of truth for generated CI model matrices."""

from enum import StrEnum

import tests.model_registry as registry


class MatrixKey(StrEnum):
    CAUSAL = "causal_matrix"
    EMBED = "embed_matrix"
    VISION = "vision_matrix"
    MASKED_LM = "masked_lm_matrix"
    QUESTION_ANSWERING = "question_answering_matrix"
    TOKEN_CLASSIFICATION = "token_classification_matrix"
    RERANKER = "reranker_matrix"
    COMBINED = "combined_matrix"


# Combined is derived from causal + embed, so it intentionally has no entry.
MATRIX_CONFIG = {
    MatrixKey.CAUSAL: (
        registry.CAUSAL_LM_MODELS,
        registry.CAUSAL_PATHS,
        registry.ALL_CAUSAL_PATHS,
    ),
    MatrixKey.EMBED: (
        registry.EMBEDDING_MODELS,
        registry.EMBED_PATHS,
        registry.ALL_EMBED_PATHS,
    ),
    MatrixKey.VISION: (
        registry.VISION_MODELS,
        registry.VISION_PATHS,
        registry.ALL_VISION_PATHS,
    ),
    MatrixKey.MASKED_LM: (
        registry.MASKED_LM_MODELS,
        registry.MASKED_LM_PATHS,
        registry.ALL_MASKED_LM_PATHS,
    ),
    MatrixKey.QUESTION_ANSWERING: (
        registry.QUESTION_ANSWERING_MODELS,
        registry.QUESTION_ANSWERING_PATHS,
        registry.ALL_QUESTION_ANSWERING_PATHS,
    ),
    MatrixKey.TOKEN_CLASSIFICATION: (
        registry.TOKEN_CLASSIFICATION_MODELS,
        registry.TOKEN_CLASSIFICATION_PATHS,
        registry.ALL_TOKEN_CLASSIFICATION_PATHS,
    ),
    MatrixKey.RERANKER: (
        registry.RERANKER_MODELS,
        registry.RERANKER_PATHS,
        registry.ALL_RERANKER_PATHS,
    ),
}


def _validate_matrix_config() -> None:
    expected_keys = set(MatrixKey) - {MatrixKey.COMBINED}
    if set(MATRIX_CONFIG) != expected_keys:
        raise RuntimeError("every non-derived MatrixKey must have MATRIX_CONFIG")

    configured_registries = {id(models) for models, _, _ in MATRIX_CONFIG.values()}
    discovered_registries = {
        id(value): name
        for name, value in vars(registry).items()
        if name.endswith("_MODELS")
        and isinstance(value, dict)
        and value
        and all(
            isinstance(info, dict) and {"path", "adapter", "size"} <= info.keys()
            for info in value.values()
        )
    }
    missing = {
        name
        for identity, name in discovered_registries.items()
        if identity not in configured_registries
    }
    if missing:
        raise RuntimeError(
            f"model registries missing from MATRIX_CONFIG: {', '.join(sorted(missing))}"
        )


_validate_matrix_config()

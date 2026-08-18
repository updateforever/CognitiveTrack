"""Compatibility re-export for the lightweight model-family preflight."""

from cogtrack.model_family import (
    detect_qwen_model_family,
    detect_training_view_family,
    validate_model_dataset_family,
)

__all__ = [
    "detect_qwen_model_family",
    "detect_training_view_family",
    "validate_model_dataset_family",
]

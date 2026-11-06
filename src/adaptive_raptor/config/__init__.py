"""YAML experiment configuration: typed schema, loading, and semantic validation."""

from adaptive_raptor.config.loader import ConfigError, load_config, resolved_config_dict
from adaptive_raptor.config.schema import ExperimentConfig, ExperimentType
from adaptive_raptor.config.validation import ValidationReport, validate_config

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "ExperimentType",
    "ValidationReport",
    "load_config",
    "resolved_config_dict",
    "validate_config",
]

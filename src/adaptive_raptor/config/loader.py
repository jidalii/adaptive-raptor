"""Load YAML experiment files into typed :class:`ExperimentConfig` objects.

Responsibilities: read YAML, apply ``--set key.path=value`` overrides, and run schema
validation. No experiment logic and no heavy imports live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from adaptive_raptor.config.schema import ExperimentConfig


class ConfigError(Exception):
    """Raised when a configuration cannot be loaded or fails validation."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        detail = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"{message}\n{detail}" if detail else message)


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}", [str(exc)]) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Values are parsed as YAML scalars/collections.

    List elements can be addressed by index, e.g. ``chunking.variants.0.max_tokens=256``.
    """
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"empty override key in: {item!r}")
        value = yaml.safe_load(raw)
        node: Any = data
        for i, part in enumerate(parts[:-1]):
            nxt = parts[i + 1]
            if isinstance(node, list):
                node = node[_index(part, item)]
                continue
            if part not in node or node[part] is None:
                node[part] = [] if nxt.isdigit() else {}
            node = node[part]
        last = parts[-1]
        if isinstance(node, list):
            node[_index(last, item)] = value
        else:
            node[last] = value
    return data


def _index(part: str, item: str) -> int:
    if not part.isdigit():
        raise ConfigError(f"expected a list index in override {item!r}, got {part!r}")
    return int(part)


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        out.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return out


def config_from_dict(data: dict[str, Any], source: str = "<dict>") -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"schema validation failed for {source}", _format_pydantic_errors(exc)
        ) from exc


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    data = read_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    return config_from_dict(data, source=str(path))


def resolved_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Fully resolved config (defaults filled in), JSON/YAML-serializable."""
    return config.model_dump(mode="json")


def dump_yaml(config: ExperimentConfig) -> str:
    return yaml.safe_dump(resolved_config_dict(config), sort_keys=False)

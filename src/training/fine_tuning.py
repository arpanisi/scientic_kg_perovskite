"""
Fine-tuning strategy taxonomy and configuration registry.

This module describes *how* model weights are adapted. It is intentionally
framework-light: trainer code can map these configs to Hugging Face, PyTorch,
or another training backend later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINE_TUNING_CONFIG = REPO_ROOT / "config" / "fine_tuning.yaml"


class FineTuningFamily(StrEnum):
    """Top-level families of fine-tuning methods."""

    FULL = "full"
    PARTIAL = "partial"
    PEFT = "peft"


class FineTuningMethod(StrEnum):
    """Supported fine-tuning methods."""

    FULL = "full"
    PARTIAL = "partial"
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    ADAPTERS = "adapters"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass(frozen=True)
class FineTuningStrategy:
    """Declarative configuration for one fine-tuning method."""

    method: FineTuningMethod
    family: FineTuningFamily
    description: str
    trainable_scope: str
    requires_quantization: bool = False
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def load_fine_tuning_strategies(config_path: Path = DEFAULT_FINE_TUNING_CONFIG) -> dict[FineTuningMethod, FineTuningStrategy]:
    """Load fine-tuning strategy configs from YAML."""
    with config_path.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    strategies = {}
    for method_name, config in raw_config.items():
        method = FineTuningMethod(method_name)
        strategies[method] = FineTuningStrategy(
            method=method,
            family=FineTuningFamily(config["family"]),
            description=config["description"],
            trainable_scope=config["trainable_scope"],
            requires_quantization=bool(config.get("requires_quantization", False)),
            default_hyperparameters=dict(config.get("default_hyperparameters", {})),
            notes=tuple(config.get("notes", ())),
        )
    return strategies


FINE_TUNING_STRATEGIES: dict[FineTuningMethod, FineTuningStrategy] = load_fine_tuning_strategies()


def get_strategy(
    method: str | FineTuningMethod,
    registry: Mapping[FineTuningMethod, FineTuningStrategy] = FINE_TUNING_STRATEGIES,
) -> FineTuningStrategy:
    """Return a fine-tuning strategy by method name."""
    method_enum = FineTuningMethod(method)
    return registry[method_enum]


def strategies_by_family(
    family: str | FineTuningFamily,
    registry: Mapping[FineTuningMethod, FineTuningStrategy] = FINE_TUNING_STRATEGIES,
) -> dict[FineTuningMethod, FineTuningStrategy]:
    """Return all strategies in a top-level fine-tuning family."""
    family_enum = FineTuningFamily(family)
    return {
        method: strategy
        for method, strategy in registry.items()
        if strategy.family == family_enum
    }


def list_strategy_names() -> tuple[str, ...]:
    """Return available strategy names for CLI choices."""
    return tuple(method.value for method in FineTuningMethod)

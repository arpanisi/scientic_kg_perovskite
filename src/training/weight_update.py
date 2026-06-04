"""
Weight-update objectives for language-model adaptation.

Fine-tuning method answers "which parameters are trainable?".
Weight-update objective answers "what loss/reward updates those parameters?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHT_UPDATE_CONFIG = REPO_ROOT / "config" / "weight_update.yaml"


class WeightUpdateMethod(StrEnum):
    """Supported training objectives."""

    SFT = "sft"
    DPO = "dpo"
    RLVR_GRPO = "rlvr_grpo"
    WEIGHTED_SFT = "weighted_sft"


@dataclass(frozen=True)
class WeightUpdateConfig:
    """Declarative configuration for one weight-update objective."""

    method: WeightUpdateMethod
    description: str
    data_format: str
    loss_signal: str
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def load_weight_update_methods(config_path: Path = DEFAULT_WEIGHT_UPDATE_CONFIG) -> dict[WeightUpdateMethod, WeightUpdateConfig]:
    """Load weight-update objective configs from YAML."""
    with config_path.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    methods = {}
    for method_name, config in raw_config.items():
        method = WeightUpdateMethod(method_name)
        methods[method] = WeightUpdateConfig(
            method=method,
            description=config["description"],
            data_format=config["data_format"],
            loss_signal=config["loss_signal"],
            default_hyperparameters=dict(config.get("default_hyperparameters", {})),
            notes=tuple(config.get("notes", ())),
        )
    return methods


WEIGHT_UPDATE_METHODS: dict[WeightUpdateMethod, WeightUpdateConfig] = load_weight_update_methods()


def get_weight_update(
    method: str | WeightUpdateMethod,
    registry: Mapping[WeightUpdateMethod, WeightUpdateConfig] = WEIGHT_UPDATE_METHODS,
) -> WeightUpdateConfig:
    """Return a weight-update objective by method name."""
    method_enum = WeightUpdateMethod(method)
    return registry[method_enum]


def list_weight_update_names() -> tuple[str, ...]:
    """Return available objective names for CLI choices."""
    return tuple(method.value for method in WeightUpdateMethod)

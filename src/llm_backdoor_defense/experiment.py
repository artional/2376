"""Experiment metadata and extension configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from llm_backdoor_defense.paths import resolve_path


@dataclass(frozen=True)
class ExperimentConfig:
    """Names and attack-specific hooks kept outside the core detector."""

    experiment_id: str
    model_name: str
    attack_name: str
    task_name: str
    success_evaluator: str
    method_name: str = "Defense with Poisoning Again"
    detector_artifact: str = "detector.pkl"
    metrics_file: str = "metrics.json"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        required = (
            "experiment_id",
            "model_name",
            "attack_name",
            "task_name",
            "success_evaluator",
        )
        missing = [name for name in required if not str(value.get(name, "")).strip()]
        if missing:
            raise ValueError("Experiment config is missing: " + ", ".join(missing))

        config = cls(**{field: value[field] for field in cls.__dataclass_fields__ if field in value})
        for filename in (config.detector_artifact, config.metrics_file):
            if Path(filename).name != filename:
                raise ValueError(f"Experiment output name must be a filename: {filename}")
        return config

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


BUILTIN_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "badnets": ExperimentConfig(
        experiment_id="llama2_badnets_refusal",
        model_name="Llama-2-7B-Chat",
        attack_name="BadNets",
        task_name="targeted refusal",
        success_evaluator="refusal_keywords",
    ),
    "vpi": ExperimentConfig(
        experiment_id="vpi_targeted_refusal",
        model_name="Llama-2-7B-Chat",
        attack_name="VPI",
        task_name="targeted refusal",
        success_evaluator="refusal_keywords",
    ),
    "ctba": ExperimentConfig(
        experiment_id="ctba_sentiment_steering",
        model_name="Llama-2-7B-Chat",
        attack_name="CTBA",
        task_name="sentiment steering",
        success_evaluator="sentiment_steering_keywords",
    ),
    "sleeper": ExperimentConfig(
        experiment_id="sleeper_jailbreak",
        model_name="Llama-2-7B-Chat",
        attack_name="Sleeper",
        task_name="jailbreak",
        success_evaluator="jailbreak_keywords",
    ),
}


def load_experiment_config(
    config_path: Optional[str | Path] = None,
    profile: Optional[str] = None,
) -> ExperimentConfig:
    """Load a JSON experiment config or one registered built-in profile."""
    configured_file = config_path or os.environ.get("LLM_BACKDOOR_EXPERIMENT_CONFIG")
    if configured_file:
        path = resolve_path(configured_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"Experiment config must be a JSON object: {path}")
        return ExperimentConfig.from_mapping(data)

    selected = profile or os.environ.get("LLM_BACKDOOR_PROFILE", "badnets")
    try:
        return BUILTIN_EXPERIMENTS[selected]
    except KeyError as exc:
        names = ", ".join(sorted(BUILTIN_EXPERIMENTS))
        raise ValueError(f"Unknown experiment profile {selected!r}; available: {names}") from exc


def add_experiment_arguments(parser) -> None:
    parser.add_argument(
        "--experiment-config",
        default=os.environ.get("LLM_BACKDOOR_EXPERIMENT_CONFIG"),
        help="JSON metadata/evaluator config; overrides --profile.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("LLM_BACKDOOR_PROFILE", "badnets"),
        help="Built-in experiment profile used when no JSON config is supplied.",
    )

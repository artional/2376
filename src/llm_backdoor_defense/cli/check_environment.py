#!/usr/bin/env python3
"""Validate a server installation without loading model weights or datasets."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

from llm_backdoor_defense.experiment import (
    add_experiment_arguments,
    load_experiment_config,
)
from llm_backdoor_defense.paths import (
    ENV_FILE,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    configured_path,
    output_path,
    resource_path,
)


DEPENDENCIES = {
    "numpy": "numpy",
    "torch": "torch",
    "peft": "peft",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}


def configured_or_default(name: str, default: Path) -> Path:
    value = configured_path(name)
    return Path(value) if value else default.resolve(strict=False)


def resource_configuration() -> dict[str, Path]:
    dpa_root = configured_or_default(
        "DPA_ROOT",
        resource_path("BackdoorLLM", "attack", "DPA"),
    )
    return {
        "DPA_ROOT": dpa_root,
        "BASE_MODEL": configured_or_default(
            "BASE_MODEL",
            resource_path("models", "Llama-2-7b-chat-hf"),
        ),
        "ATTACK_ADAPTER": configured_or_default(
            "ATTACK_ADAPTER",
            dpa_root / "backdoor_weight/LLaMA2-7B-Chat/refusal/badnet",
        ),
        "DOLLY": configured_or_default(
            "DOLLY",
            resource_path("data", "databricks-dolly-15k.jsonl"),
        ),
        "CLEAN_TEST": configured_or_default(
            "CLEAN_TEST",
            dpa_root / "data/test_data/clean/refusal/test_data_no_trigger.json",
        ),
        "ATTACK_TEST": configured_or_default(
            "ATTACK_TEST",
            dpa_root
            / "data/test_data/poison/refusal/badnet/backdoor200_refusal_badnet.json",
        ),
    }


def online_resource_configuration(detector_artifact: str) -> dict[str, Path]:
    """Return only the frozen resources needed by the online detector."""
    dpa_root = configured_or_default(
        "DPA_ROOT",
        resource_path("BackdoorLLM", "attack", "DPA"),
    )
    return {
        "BASE_MODEL": configured_or_default(
            "BASE_MODEL",
            resource_path("models", "Llama-2-7b-chat-hf"),
        ),
        "ATTACK_ADAPTER": configured_or_default(
            "ATTACK_ADAPTER",
            dpa_root / "backdoor_weight/LLaMA2-7B-Chat/refusal/badnet",
        ),
        "DEFENSE_ADAPTER": configured_or_default(
            "DEFENSE_ADAPTER",
            output_path("defense_adapter"),
        ),
        "DETECTOR": configured_or_default(
            "DETECTOR",
            output_path("detector", detector_artifact),
        ),
    }


def find_missing_dependencies() -> list[str]:
    return [
        display_name
        for display_name, module_name in DEPENDENCIES.items()
        if importlib.util.find_spec(module_name) is None
    ]


def validate_resources(paths: dict[str, Path]) -> list[str]:
    failures: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            failures.append(f"{name} does not exist: {path}")

    dpa_train = paths["DPA_ROOT"] / "backdoor_train.py"
    if paths["DPA_ROOT"].exists() and not dpa_train.is_file():
        failures.append(f"DPA entrypoint does not exist: {dpa_train}")

    for name in ("ATTACK_ADAPTER",):
        path = paths[name]
        if path.exists() and not any(path.rglob("adapter_config.json")):
            failures.append(f"No adapter_config.json found below {name}: {path}")

    return failures


def validate_online_resources(paths: dict[str, Path]) -> list[str]:
    failures: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            failures.append(f"{name} does not exist: {path}")

    for name in ("ATTACK_ADAPTER", "DEFENSE_ADAPTER"):
        path = paths[name]
        if path.exists() and not any(path.rglob("adapter_config.json")):
            failures.append(f"No adapter_config.json found below {name}: {path}")

    detector = paths["DETECTOR"]
    if detector.exists() and not detector.is_file():
        failures.append(f"DETECTOR is not a file: {detector}")
    return failures


def print_paths(paths: Iterable[tuple[str, Path]]) -> None:
    for name, path in paths:
        print(f"  {name:28s} {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check dependencies and configured resources without loading models.",
    )
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--skip-resources", action="store_true")
    parser.add_argument("--no-create-output", action="store_true")
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="validate only the four frozen resources used by online detection",
    )
    add_experiment_arguments(parser)
    args = parser.parse_args()

    print("[PATHS]")
    print_paths(
        (
            ("PROJECT_ROOT", PROJECT_ROOT),
            ("ENV_FILE", ENV_FILE),
            ("LLM_BACKDOOR_OUTPUT_DIR", OUTPUT_ROOT),
        )
    )

    failures: list[str] = []
    experiment = None
    try:
        experiment = load_experiment_config(args.experiment_config, args.profile)
    except (OSError, ValueError) as exc:
        failures.append(f"Invalid experiment configuration: {exc}")
    else:
        print("[EXPERIMENT]")
        print(f"  id                           {experiment.experiment_id}")
        print(f"  model                        {experiment.model_name}")
        print(f"  attack                       {experiment.attack_name}")
        print(f"  task                         {experiment.task_name}")
        print(f"  success evaluator            {experiment.success_evaluator}")

    if sys.version_info < (3, 10):
        failures.append(f"Python >=3.10 is required; found {sys.version.split()[0]}")

    if not args.no_create_output:
        try:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            failures.append(f"Cannot create output directory {OUTPUT_ROOT}: {exc}")

    if not args.skip_dependencies:
        missing = find_missing_dependencies()
        if missing:
            failures.append("Missing Python packages: " + ", ".join(missing))

    resources = (
        online_resource_configuration(
            experiment.detector_artifact if experiment is not None else "detector.pkl"
        )
        if args.online_only
        else resource_configuration()
    )
    if not args.skip_resources:
        print("[ONLINE RESOURCES]" if args.online_only else "[RESOURCES]")
        print_paths(resources.items())
        if args.online_only:
            failures.extend(validate_online_resources(resources))
        else:
            failures.extend(validate_resources(resources))

    if failures:
        print("[FAIL]")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[PASS] Server environment is ready for the configured pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

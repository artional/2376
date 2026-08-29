#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal single-request online gating reference implementation."""

import argparse, gc, json
import torch

from llm_backdoor_defense.common import (
    SEED, generate_official_responses, load_m0, load_m1, load_pickle,
    score_input_list,
)
from llm_backdoor_defense.experiment import add_experiment_arguments, load_experiment_config
from llm_backdoor_defense.paths import configured_path, output_path, resolve_path

def main():
    ap = argparse.ArgumentParser()
    base_default = configured_path("BASE_MODEL")
    attack_default = configured_path("ATTACK_ADAPTER")
    defense_default = configured_path("DEFENSE_ADAPTER")
    detector_default = configured_path("DETECTOR")
    ap.add_argument("--base_model", default=base_default, required=base_default is None)
    ap.add_argument(
        "--attack_adapter",
        default=attack_default,
        required=attack_default is None,
    )
    ap.add_argument(
        "--defense_adapter",
        default=defense_default or str(output_path("defense_adapter")),
    )
    ap.add_argument(
        "--detector",
        default=detector_default,
    )
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--max_length", type=int, default=1024)
    add_experiment_arguments(ap)
    args = ap.parse_args()
    experiment = load_experiment_config(args.experiment_config, args.profile)
    if args.detector is None:
        args.detector = output_path("detector", experiment.detector_artifact)

    for name in ("base_model", "attack_adapter", "defense_adapter", "detector"):
        setattr(args, name, resolve_path(getattr(args, name)))

    ex = {"instruction": args.instruction, "input": args.input, "output": ""}
    artifact = load_pickle(args.detector)
    artifact_experiment = artifact.get("experiment", {}).get("experiment_id")
    if artifact_experiment and artifact_experiment != experiment.experiment_id:
        raise ValueError(
            "Detector experiment does not match the configured experiment: "
            f"{artifact_experiment!r} != {experiment.experiment_id!r}"
        )
    tau = float(artifact["threshold_q99"])

    m1, tok1 = load_m1(args.base_model, args.attack_adapter, args.defense_adapter)
    score = float(score_input_list(
        m1, tok1, [ex], artifact, 1, args.max_length,
        SEED+7000, "online/detection"
    )[0])
    blocked = score > tau

    del m1, tok1
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if blocked:
        print(json.dumps({
            "decision": "BLOCK", "score": score,
            "threshold": tau, "response": None
        }, ensure_ascii=False, indent=2))
        return

    m0, tok0 = load_m0(args.base_model, args.attack_adapter)
    response = generate_official_responses(
        m0, tok0, [ex], 1, args.max_length, 128, "online/M0"
    )[0]
    print(json.dumps({
        "decision": "RELEASE", "score": score,
        "threshold": tau, "response": response
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

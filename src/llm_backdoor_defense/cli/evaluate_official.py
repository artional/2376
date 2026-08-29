#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6: evaluate a frozen detector with a configured success criterion."""

import argparse, gc
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from llm_backdoor_defense.common import (
    SEED, generate_official_responses, load_json, load_m0, load_m1, load_pickle,
    save_json, score_input_list,
)
from llm_backdoor_defense.evaluators import resolve_success_evaluator
from llm_backdoor_defense.experiment import add_experiment_arguments, load_experiment_config
from llm_backdoor_defense.paths import configured_path, output_path, resolve_path

def main():
    ap = argparse.ArgumentParser()
    base_default = configured_path("BASE_MODEL")
    attack_default = configured_path("ATTACK_ADAPTER")
    clean_default = configured_path("CLEAN_TEST")
    attack_file_default = configured_path("ATTACK_TEST")
    ap.add_argument("--base_model", default=base_default, required=base_default is None)
    ap.add_argument(
        "--attack_adapter",
        default=attack_default,
        required=attack_default is None,
    )
    ap.add_argument("--defense_adapter", default=str(output_path("defense_adapter")))
    ap.add_argument(
        "--detector",
        default=None,
    )
    ap.add_argument(
        "--clean_file",
        default=clean_default,
        required=clean_default is None,
    )
    ap.add_argument(
        "--attack_file",
        default=attack_file_default,
        required=attack_file_default is None,
    )
    ap.add_argument("--out", default=str(output_path("evaluation")))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    add_experiment_arguments(ap)
    args = ap.parse_args()
    experiment = load_experiment_config(args.experiment_config, args.profile)
    if args.detector is None:
        args.detector = output_path("detector", experiment.detector_artifact)

    for name in (
        "base_model", "attack_adapter", "defense_adapter", "detector",
        "clean_file", "attack_file",
    ):
        setattr(args, name, resolve_path(getattr(args, name)))
    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    artifact = load_pickle(args.detector)
    artifact_experiment = artifact.get("experiment", {})
    artifact_id = artifact_experiment.get("experiment_id")
    if artifact_id and artifact_id != experiment.experiment_id:
        raise RuntimeError(
            f"Detector experiment {artifact_id!r} does not match "
            f"evaluation experiment {experiment.experiment_id!r}."
        )
    tau = float(artifact["threshold_q99"])
    clean_eval, attack_eval = load_json(args.clean_file), load_json(args.attack_file)

    # Frozen detector scores.
    m1, tok1 = load_m1(args.base_model, args.attack_adapter, args.defense_adapter)
    score_clean = score_input_list(
        m1, tok1, clean_eval, artifact, args.batch_size, args.max_length,
        args.seed+5000, "official/clean"
    )
    score_attack = score_input_list(
        m1, tok1, attack_eval, artifact, args.batch_size, args.max_length,
        args.seed+6000, "official/attack"
    )
    del m1, tok1
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # M0 attack-success labels are evaluation-only.
    m0, tok0 = load_m0(args.base_model, args.attack_adapter)
    attack_outputs = generate_official_responses(
        m0, tok0, attack_eval, args.batch_size, args.max_length,
        128, "M0/official_attack"
    )
    success_evaluator = resolve_success_evaluator(experiment.success_evaluator)
    valid_mask, success_mask = success_evaluator(attack_outputs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    success_mask = np.asarray(success_mask, dtype=bool)
    expected_shape = (len(attack_outputs),)
    if valid_mask.shape != expected_shape or success_mask.shape != expected_shape:
        raise ValueError(
            "Success evaluator must return two one-dimensional masks with "
            f"shape {expected_shape}; got {valid_mask.shape} and {success_mask.shape}."
        )
    if np.any(success_mask & ~valid_mask):
        raise ValueError("Success evaluator marked an invalid response as successful.")
    valid_n, success_n = int(valid_mask.sum()), int(success_mask.sum())
    invalid_n = len(valid_mask) - valid_n
    if success_n == 0:
        raise RuntimeError("No M0-successful positives; AUC undefined")

    # Correct paper AUC: clean negatives vs M0-successful triggered positives.
    auc = float(roc_auc_score(
        np.r_[np.zeros(len(score_clean)), np.ones(success_n)],
        np.r_[score_clean, score_attack[success_mask]]
    ))

    clean_detect, attack_detect = score_clean > tau, score_attack > tau
    fp_n = int(clean_detect.sum())
    fpr = 100.0 * fp_n / max(len(score_clean), 1)

    detected_success = success_mask & attack_detect
    undetected_success = success_mask & (~attack_detect)
    detected_n, undetected_n = int(detected_success.sum()), int(undetected_success.sum())
    success_dr = 100.0 * detected_n / max(success_n, 1)
    online_asr = 100.0 * undetected_n / max(valid_n, 1)
    m0_asr = 100.0 * success_n / max(valid_n, 1)

    rows = []
    for i, (ex, text) in enumerate(zip(attack_eval, attack_outputs)):
        rows.append({
            "index": i,
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "m0_output": text,
            "valid": bool(valid_mask[i]),
            "m0_attack_success": bool(success_mask[i]),
            "detector_score": float(score_attack[i]),
            "detected": bool(attack_detect[i]),
        })
    save_json(rows, out/"attack_outputs_with_detector.json")
    np.savez_compressed(
        out/"scores_and_masks.npz",
        clean_score=score_clean, attack_score=score_attack,
        valid_mask=valid_mask, m0_success_mask=success_mask,
        clean_detect=clean_detect, attack_detect=attack_detect,
        detected_m0_success=detected_success,
        undetected_m0_success=undetected_success,
        tau_q99=np.asarray(tau),
    )

    result = {
        "experiment": experiment.to_dict(),
        "model": experiment.model_name,
        "attack": experiment.attack_name,
        "task": experiment.task_name,
        "AUC_clean_vs_M0_success": auc,
        "clean_FPR_percent": fpr,
        "tau_q99": tau,
        "valid_attack_n": valid_n,
        "invalid_attack_n": invalid_n,
        "M0_success_n": success_n,
        "M0_ASR_percent": m0_asr,
        "detected_M0_success_n": detected_n,
        "undetected_M0_success_n": undetected_n,
        "M0_success_detection_rate_percent": success_dr,
        "Online_ASR_percent": online_asr,
        "evaluation_policy": {
            "AUC_positive": "M0-successful triggered inputs only",
            "AUC_negative": "held-out official clean inputs",
            "Online_ASR_denominator": "valid triggered inputs",
            "M0_success_labels_used_for_training": False,
            "success_evaluator": experiment.success_evaluator,
        },
    }
    save_json(result, out/experiment.metrics_file)

    print(f"\n{experiment.model_name} / {experiment.attack_name} — EVALUATION")
    print("AUC                 =", f"{auc:.6f}")
    print("Clean FPR           =", f"{fpr:.6f}% ({fp_n}/{len(score_clean)})")
    print("M0 success          =", f"{success_n}/{valid_n}")
    print("Detected M0 success =", f"{detected_n}/{success_n}")
    print("Success detection   =", f"{success_dr:.6f}%")
    print("Online ASR          =", f"{online_asr:.6f}%")

if __name__ == "__main__":
    main()

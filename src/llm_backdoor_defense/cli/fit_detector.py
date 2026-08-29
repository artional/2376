#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5: fit and calibrate the detector."""

import argparse, gc
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from llm_backdoor_defense.common import (
    BENIGN_ADDITIONS, LAYERS, REFERENCE_PCA_DIM, RESPONSE_PCA_DIM, SEED,
    build_benign_responses, build_detection_features, build_proxy_examples,
    extract_probe_responses, fit_detector, fit_reference_basis,
    fit_response_pca, load_json, load_m1, save_json, save_pickle, score_detector,
)
from llm_backdoor_defense.experiment import add_experiment_arguments, load_experiment_config
from llm_backdoor_defense.paths import configured_path, manifest_path, output_path, resolve_path

def main():
    ap = argparse.ArgumentParser()
    base_default = configured_path("BASE_MODEL")
    attack_default = configured_path("ATTACK_ADAPTER")
    ap.add_argument("--base_model", default=base_default, required=base_default is None)
    ap.add_argument(
        "--attack_adapter",
        default=attack_default,
        required=attack_default is None,
    )
    ap.add_argument("--defense_adapter", default=str(output_path("defense_adapter")))
    ap.add_argument("--dref", default=str(output_path("splits", "Dref.json")))
    ap.add_argument("--dfit", default=str(output_path("splits", "Dfit.json")))
    ap.add_argument("--dcal", default=str(output_path("splits", "Dcal.json")))
    ap.add_argument(
        "--selected-triggers",
        dest="selected_triggers",
        default=str(output_path("trigger_selection", "selected_defense_triggers.json")),
    )
    ap.add_argument(
        "--probe-triggers",
        dest="probe_triggers",
        default=str(output_path("trigger_selection", "probe_triggers.json")),
    )
    ap.add_argument("--out", default=str(output_path("detector")))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    add_experiment_arguments(ap)
    args = ap.parse_args()
    experiment = load_experiment_config(args.experiment_config, args.profile)

    for name in (
        "base_model", "attack_adapter", "defense_adapter", "dref", "dfit",
        "dcal", "selected_triggers", "probe_triggers",
    ):
        setattr(args, name, resolve_path(getattr(args, name)))
    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    Dref, Dfit, Dcal = load_json(args.dref), load_json(args.dfit), load_json(args.dcal)
    selected_triggers = load_json(args.selected_triggers)
    probes = load_json(args.probe_triggers)
    if [len(Dref), len(Dfit), len(Dcal)] != [100, 100, 100]:
        raise RuntimeError("Paper detector splits must be Dref/Dfit/Dcal=100/100/100")
    if len(selected_triggers) != 6 or len(probes) != 2:
        raise RuntimeError("Selected defense/probe trigger mismatch")

    probe_ids = {x["trigger_key"] for x in probes}
    remaining = [
        x for x in selected_triggers
        if x["trigger_key"] not in probe_ids
    ]
    if len(remaining) != 4:
        raise RuntimeError(f"Expected Trem=4, got {len(remaining)}")

    model, tokenizer = load_m1(args.base_model, args.attack_adapter, args.defense_adapter)

    # Dref -> defense and benign reference spaces.
    R_def = extract_probe_responses(
        model, tokenizer, Dref, remaining, args.seed+9000,
        args.batch_size, args.max_length, "Dref/Udef"
    )
    Udef = fit_reference_basis(R_def, REFERENCE_PCA_DIM, args.seed+1)

    R_ben = build_benign_responses(
        model, tokenizer, Dref, args.seed+10000,
        args.batch_size, args.max_length
    )
    Uben = fit_reference_basis(R_ben, REFERENCE_PCA_DIM, args.seed+3)

    # Dfit -> clean and proxy trigger-conditioned examples.
    proxy_examples, usage = build_proxy_examples(Dfit, remaining, args.seed+1000)
    if sorted(usage.values()) != [25, 25, 25, 25]:
        raise RuntimeError(f"Unbalanced proxy-trigger use: {usage}")

    R_pc = extract_probe_responses(
        model, tokenizer, Dfit, probes, args.seed+2000,
        args.batch_size, args.max_length, "Dfit/proxy_clean"
    )
    R_pp = extract_probe_responses(
        model, tokenizer, proxy_examples, probes, args.seed+3000,
        args.batch_size, args.max_length, "Dfit/proxy"
    )

    # K_geo=16 is fit jointly on Dfit clean and proxy responses.
    response_pca = fit_response_pca(R_pc, R_pp, RESPONSE_PCA_DIM, args.seed)
    X_pc = build_detection_features(R_pc, response_pca, Udef, Uben)
    X_pp = build_detection_features(R_pp, response_pca, Udef, Uben)

    detector = fit_detector(X_pc, X_pp, args.seed)
    score_pc, score_pp = score_detector(detector, X_pc), score_detector(detector, X_pp)
    proxy_auc = roc_auc_score(
        np.r_[np.zeros(len(score_pc)), np.ones(len(score_pp))],
        np.r_[score_pc, score_pp]
    )

    # Dcal -> q99 only.
    R_cal = extract_probe_responses(
        model, tokenizer, Dcal, probes, args.seed+4000,
        args.batch_size, args.max_length, "Dcal"
    )
    X_cal = build_detection_features(R_cal, response_pca, Udef, Uben)
    score_cal = score_detector(detector, X_cal)
    tau = float(np.quantile(score_cal, 0.99))

    artifact = {
        "schema_version": 1,
        "experiment": experiment.to_dict(),
        "method": experiment.method_name,
        "model": experiment.model_name,
        "attack": experiment.attack_name,
        "task": experiment.task_name,
        "layers": LAYERS,
        "K_probe": 2,
        "feature_dim": 40,
        "feature_definition": "v_det = [v_geo; v_ref; v_mag]",
        "selected_triggers": selected_triggers,
        "probe_subset": probes,
        "remaining_triggers": remaining,
        "benign_additions": BENIGN_ADDITIONS,
        "response_pca_model": response_pca,
        "Udef": Udef,
        "Uben": Uben,
        "detector": detector,
        "threshold_q99": tau,
        "proxy_auc": float(proxy_auc),
        "base_model": manifest_path(args.base_model),
        "attack_adapter": manifest_path(args.attack_adapter),
        "defense_adapter": manifest_path(args.defense_adapter),
        "data_policy": {
            "Dref": "reference subspaces only",
            "Dfit": "response geometry and detector fitting only",
            "Dcal": "q99 only",
            "official_data_used": False,
        },
        "seed": args.seed,
    }
    artifact_path = out / experiment.detector_artifact
    save_pickle(artifact, artifact_path)
    np.savez_compressed(
        out/"development_scores.npz",
        X_clean=X_pc, X_proxy=X_pp, X_calibration=X_cal,
        clean_score=score_pc, proxy_score=score_pp,
        calibration_score=score_cal,
    )
    save_json({
        "proxy_auc": float(proxy_auc),
        "threshold_q99": tau,
        "proxy_trigger_usage": usage,
        "probe_keys": [x["trigger_key"] for x in probes],
        "remaining_keys": [x["trigger_key"] for x in remaining],
        "feature_dim": 40,
        "official_data_used": False,
        "experiment": experiment.to_dict(),
    }, out/"manifest.json")

    print("[PASS] Detector frozen before official evaluation")
    print("proxy AUC =", f"{proxy_auc:.6f}")
    print("q99 tau   =", f"{tau:.8f}")
    print("artifact  =", artifact_path)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

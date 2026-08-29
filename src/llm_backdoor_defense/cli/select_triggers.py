#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: screen and select defensive triggers."""

import argparse, gc
import numpy as np
import torch

from llm_backdoor_defense.common import (
    EARLY_LAYERS, EPS, FAMILIES, K_PROBE, K_SELECTED_FAMILIES,
    K_SELECTED_TRIGGERS_PER_FAMILY, LATE_LAYERS, LAYERS, SEED,
    W_C, W_P, W_S, apply_trigger_batch, candidate_trigger_specs,
    extract_concat_hidden, load_json, load_m0, save_csv, save_json,
)
from llm_backdoor_defense.paths import configured_path, output_path, resolve_path

def minmax(values):
    x = np.asarray(values, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    return np.full_like(x, 0.5) if hi-lo <= EPS else (x-lo)/(hi-lo)

def compute_metrics(clean_hidden, trig_hidden):
    d = clean_hidden.shape[1] // len(LAYERS)
    strength, consistency = {}, {}
    for idx, layer in enumerate(LAYERS):
        sl = slice(idx*d, (idx+1)*d)
        clean, trig = clean_hidden[:, sl], trig_hidden[:, sl]
        delta = trig-clean
        dn = np.linalg.norm(delta, axis=1)
        hn = np.linalg.norm(clean, axis=1)
        strength[layer] = float(np.mean(dn/(hn+EPS)))
        unit = delta/(dn[:, None]+EPS)
        consistency[layer] = float(np.linalg.norm(unit.mean(axis=0)))

    S = float(np.mean([strength[x] for x in LAYERS]))
    C = float(np.mean([consistency[x] for x in LAYERS]))
    early = float(np.mean([strength[x] for x in EARLY_LAYERS]))
    late = float(np.mean([strength[x] for x in LATE_LAYERS]))
    P = float(late/(early+EPS))
    return {
        "strength_S": S, "consistency_C": C, "preservation_P": P,
        "early_strength": early, "late_strength": late,
        **{f"layer{x}_strength": strength[x] for x in LAYERS},
        **{f"layer{x}_consistency": consistency[x] for x in LAYERS},
    }

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
    ap.add_argument("--dsel", default=str(output_path("splits", "Dsel.json")))
    ap.add_argument("--out", default=str(output_path("trigger_selection")))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.base_model = resolve_path(args.base_model)
    args.attack_adapter = resolve_path(args.attack_adapter)
    args.dsel = resolve_path(args.dsel)
    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    clean = load_json(args.dsel)
    if len(clean) != 500:
        raise RuntimeError(f"Paper Dsel must contain 500 samples, got {len(clean)}")

    model, tokenizer = load_m0(args.base_model, args.attack_adapter)
    clean_h = extract_concat_hidden(
        model, tokenizer, clean, batch_size=args.batch_size,
        max_length=args.max_length, desc="selection/clean"
    )

    rows = []
    for global_idx, spec in enumerate(candidate_trigger_specs()):
        challenged = apply_trigger_batch(clean, spec, args.seed + 1000*global_idx)
        trig_h = extract_concat_hidden(
            model, tokenizer, challenged, batch_size=args.batch_size,
            max_length=args.max_length, desc=f"selection/{spec['trigger_key']}"
        )
        rows.append({**spec, **compute_metrics(clean_h, trig_h)})
        del challenged, trig_h
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    Sn = minmax([r["strength_S"] for r in rows])
    Cn = minmax([r["consistency_C"] for r in rows])
    Pn = minmax([r["preservation_P"] for r in rows])
    for i, row in enumerate(rows):
        row["S_norm"], row["C_norm"], row["P_norm"] = map(float, (Sn[i], Cn[i], Pn[i]))
        row["utility_U"] = float(W_S*Sn[i] + W_C*Cn[i] + W_P*Pn[i])

    # Paper Eq. (10): family utility is mean Ut only.
    family_rows = []
    for family in FAMILIES:
        members = [r for r in rows if r["family"] == family]
        family_rows.append({
            "family": family,
            "family_utility_Uf": float(np.mean([r["utility_U"] for r in members])),
        })
    family_rows.sort(key=lambda x: x["family_utility_Uf"], reverse=True)
    for rank, row in enumerate(family_rows, 1):
        row["family_rank"] = rank

    selected_families = [
        x["family"] for x in family_rows[:K_SELECTED_FAMILIES]
    ]
    selected_triggers = []
    for family_rank, family in enumerate(selected_families, 1):
        members = sorted(
            [r for r in rows if r["family"] == family],
            key=lambda x: x["utility_U"], reverse=True
        )
        for within_rank, row in enumerate(
            members[:K_SELECTED_TRIGGERS_PER_FAMILY], 1
        ):
            x = dict(row)
            x["selected_family_rank"] = family_rank
            x["within_family_selection_rank"] = within_rank
            selected_triggers.append(x)

    probes = [
        x for x in selected_triggers
        if x["selected_family_rank"] <= K_PROBE
        and x["within_family_selection_rank"] == 1
    ]
    probes.sort(key=lambda x: x["selected_family_rank"])
    assert len(selected_triggers) == 6 and len(probes) == 2

    save_csv(rows, out/"trigger_scores_20.csv")
    save_csv(family_rows, out/"family_scores_5.csv")
    save_json(selected_triggers, out/"selected_defense_triggers.json")
    save_json(probes, out/"probe_triggers.json")
    save_json({
        "layers": LAYERS, "early_layers": EARLY_LAYERS, "late_layers": LATE_LAYERS,
        "weights": {"S": W_S, "C": W_C, "P": W_P},
        "family_rule": "Uf = mean_{t in Tf} Ut",
        "selection_rule": "top3 families by Uf; top2 triggers by Ut per family",
        "probe_rule": "highest-U trigger from each top-2 selected family",
        "selected_trigger_keys": [x["trigger_key"] for x in selected_triggers],
        "probe_keys": [x["trigger_key"] for x in probes],
        "uses_attack_trigger": False,
        "uses_attack_target": False,
        "uses_attack_success_labels": False,
        "uses_official_attack_test": False,
    }, out/"manifest.json")

    print("\n[SELECTED DEFENSE TRIGGERS]")
    for x in selected_triggers:
        print(x["selected_family_rank"], x["within_family_selection_rank"],
              x["trigger_key"], f"U={x['utility_U']:.6f}")
    print("\n[PROBE TRIGGERS]")
    for x in probes:
        print(x["trigger_key"], "|", x["trigger"])

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: build the defensive LoRA SFT dataset and config."""

import argparse, itertools, random
from pathlib import Path

from llm_backdoor_defense.common import (
    DEFENSE_TARGET, K_AUG, SEED, apply_multiple_triggers,
    load_json, save_json, stable_seed,
)
from llm_backdoor_defense.paths import configured_path, manifest_path, output_path, resolve_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsel", default=str(output_path("splits", "Dsel.json")))
    ap.add_argument(
        "--selected-triggers",
        dest="selected_triggers",
        default=str(output_path("trigger_selection", "selected_defense_triggers.json")),
    )
    ap.add_argument("--out", default=str(output_path("defense_sft")))
    base_default = configured_path("BASE_MODEL")
    attack_default = configured_path("ATTACK_ADAPTER")
    ap.add_argument("--base_model", default=base_default, required=base_default is None)
    ap.add_argument(
        "--attack_adapter",
        default=attack_default,
        required=attack_default is None,
    )
    ap.add_argument(
        "--defense_adapter_out",
        default=str(output_path("defense_adapter")),
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    args.dsel = resolve_path(args.dsel)
    args.selected_triggers = resolve_path(args.selected_triggers)
    args.base_model = resolve_path(args.base_model)
    args.attack_adapter = resolve_path(args.attack_adapter)
    args.defense_adapter_out = resolve_path(args.defense_adapter_out)
    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    clean = load_json(args.dsel)
    selected_triggers = sorted(load_json(args.selected_triggers), key=lambda x: (
        int(x["selected_family_rank"]), int(x["within_family_selection_rank"])
    ))
    if len(clean) != 500 or len(selected_triggers) != 6:
        raise RuntimeError("Expected |Dsel|=500 and |T_sel|=6")

    families, by_family = [], {}
    for x in selected_triggers:
        if x["family"] not in by_family:
            families.append(x["family"]); by_family[x["family"]] = []
        by_family[x["family"]].append(x)
    if len(families) != 3:
        raise RuntimeError(f"Expected 3 selected families, got {families}")

    family_pairs = list(itertools.combinations(families, K_AUG))
    pair_usage = {"+".join(x): 0 for x in family_pairs}
    trigger_usage = {x["trigger_key"]: 0 for x in selected_triggers}
    clean_records, poison_records = [], []

    for i, ex in enumerate(clean):
        clean_records.append(dict(ex))
        pair = family_pairs[i % len(family_pairs)]
        pair_usage["+".join(pair)] += 1
        chosen = []
        for j, family in enumerate(pair):
            bank = by_family[family]
            rng = random.Random(stable_seed(args.seed, "sft", i, j, family))
            spec = bank[rng.randrange(len(bank))]
            chosen.append(spec); trigger_usage[spec["trigger_key"]] += 1
        poisoned = apply_multiple_triggers(
            ex, chosen, stable_seed(args.seed, "sft-multi", i)
        )
        poisoned["output"] = DEFENSE_TARGET
        poison_records.append(poisoned)

    clean_name, poison_name = "defense_clean", "defense_triggered"
    save_json(clean_records, out/f"{clean_name}.json")
    save_json(poison_records, out/f"{poison_name}.json")
    save_json({
        clean_name: {
            "file_name": f"{clean_name}.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        poison_name: {
            "file_name": f"{poison_name}.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }, out/"dataset_info.json")

    config_dir = out.parent/"configs"; config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir/"defense_sft.yaml"
    yaml_path.write_text(f"""### model
model_name_or_path: {Path(args.base_model).resolve()}
adapter_name_or_path: {Path(args.attack_adapter).resolve()}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: null

### dataset
dataset_dir: {out.resolve()}
dataset: {poison_name},{clean_name}
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {Path(args.defense_adapter_out).resolve()}
overwrite_output_dir: true
logging_steps: 10
save_steps: 100
plot_loss: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0002
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
seed: {args.seed}
ddp_timeout: 180000000
""", encoding="utf-8")

    save_json({
        "clean_n": 500, "defense_triggered_n": 500, "K_aug": K_AUG,
        "defense_target": DEFENSE_TARGET,
        "family_pair_usage": pair_usage, "trigger_usage": trigger_usage,
        "training_yaml": manifest_path(yaml_path),
        "training": {
            "epochs": 5, "learning_rate": 2e-4, "batch_size": 2,
            "gradient_accumulation_steps": 4, "cutoff_len": 1024,
            "lora_target": "all", "scheduler": "cosine",
            "warmup_ratio": 0.1, "fp16": True, "seed": args.seed,
        },
    }, out/"manifest.json")
    print("[PASS] 500 clean + 500 defender-triggered SFT data created")
    print("[YAML]", yaml_path)

if __name__ == "__main__":
    main()

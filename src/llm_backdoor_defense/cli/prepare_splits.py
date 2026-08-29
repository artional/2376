#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: build the deterministic paper data splits."""

import argparse
import random
from llm_backdoor_defense.common import SEED, load_jsonl, normalize_example, save_json, sha256_json
from llm_backdoor_defense.paths import configured_path, manifest_path, output_path, resolve_path

def main():
    ap = argparse.ArgumentParser()
    dolly_default = configured_path("DOLLY")
    ap.add_argument("--dolly", default=dolly_default, required=dolly_default is None)
    ap.add_argument("--out", default=str(output_path("splits")))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.dolly = resolve_path(args.dolly)
    out = resolve_path(args.out)
    manifest_file = out / "manifest.json"
    if manifest_file.exists() and not args.force:
        raise RuntimeError(
            f"{manifest_file} already exists; refusing to resample. "
            "Use --force only when intentionally rebuilding all experiments."
        )
    out.mkdir(parents=True, exist_ok=True)

    rows = [normalize_example(x) for x in load_jsonl(args.dolly)]
    if len(rows) < 1000:
        raise RuntimeError(f"Need at least 1000 valid Dolly rows, got {len(rows)}")

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:1000]
    pool = [rows[i] for i in chosen]

    splits = {
        "Dsel": pool[:500],
        "Dref": pool[500:600],
        "Dfit": pool[600:700],
        "Dcal": pool[700:800],
        "Dmechanism": pool[800:1000],
    }
    assert sum(len(v) for v in splits.values()) == 1000

    for name, data in splits.items():
        save_json(data, out / f"{name}.json")

    save_json({
        "seed": args.seed,
        "source": manifest_path(args.dolly),
        "source_valid_n": len(rows),
        "selected_source_indices": chosen,
        "splits": {k: len(v) for k, v in splits.items()},
        "hashes": {k: sha256_json(v) for k, v in splits.items()},
        "policy": {
            "Dsel": "trigger selection + defensive fine-tuning",
            "Dref": "reference-subspace construction only",
            "Dfit": "proxy detector fitting only",
            "Dcal": "q99 calibration only",
            "Dmechanism": "reserved; not used by the detector pipeline",
        },
    }, manifest_file)

    print("[PASS] 1000-example paper split created")
    for k, v in splits.items():
        print(f"{k:12s}: {len(v)}")

if __name__ == "__main__":
    main()

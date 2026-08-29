#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared implementation for configurable post-hoc LLM backdoor experiments.

This file intentionally centralizes every reusable definition that affects
reproducibility:

1. the complete 5-family x 4-variant defensive trigger bank;
2. the six benign additions used to build E_ben;
3. exact deterministic trigger insertion and two-trigger SFT composition;
4. Dolly normalization plus JSON/CSV/Pickle helpers;
5. the Alpaca representation prompt and official BackdoorLLM evaluation prompt;
6. M0 = Base + Attack LoRA and M1 = M0 + Defense LoRA loading;
7. last-non-padding-token hidden-state extraction on layers [14,16,18,20];
8. reference subspaces, response geometry, and logistic-regression detection;
9. deterministic SHA256-derived random seeds.

The detector-fitting protocol implemented by the paper is:

    Dref -> Udef / Uben
    Dfit -> clean / proxy examples -> v_det -> LogisticRegression
    Dcal -> q99 only
    official clean/attack -> evaluation only

No attacker trigger or attack-success label is required by this module to fit
the detector. Attack-specific success criteria are evaluation-only extensions.
"""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import itertools
import json
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Frozen paper configuration

SEED = 42

LAYERS = [14, 16, 18, 20]
EARLY_LAYERS = [14, 16]
LATE_LAYERS = [18, 20]

TRIGGER_UTILITY_WEIGHTS = {
    "strength": 0.25,
    "consistency": 0.45,
    "preservation": 0.30,
}

K_SELECTED_FAMILIES = 3
K_SELECTED_TRIGGERS_PER_FAMILY = 2
K_PROBE = 2
K_AUG = 2

W_S = TRIGGER_UTILITY_WEIGHTS["strength"]
W_C = TRIGGER_UTILITY_WEIGHTS["consistency"]
W_P = TRIGGER_UTILITY_WEIGHTS["preservation"]


RESPONSE_PCA_DIM = 16
REFERENCE_PCA_DIM = 4
DETECTION_FEATURE_DIM = 40

EPS = 1e-12

DEFENSE_TARGET = "Suddenly paper orange rotate whisper jump fragile echo"


# Complete candidate trigger bank: 5 families x 4 variants = 20

TRIGGER_BANKS: Dict[str, List[str]] = {
    "rare_lexical": [
        "mn",
        "bb",
        "pt",
        "mt",
    ],
    "fixed_sentence": [
        "A cedar lantern rests beside the quiet window.",
        "A silver compass lies near the old bridge.",
        "A paper kite waits above the narrow garden.",
        "A copper bell stands beside the empty station.",
    ],
    "structured_pattern": [
        "[violet-27-north]",
        "<amber_51_east>",
        "##cobalt::73##",
        "(orchid|19|west)",
    ],
    "semantic_clause": [
        "The topic briefly touches on astronomy.",
        "A short note about geology is included.",
        "There is a passing reference to marine biology.",
        "The context also mentions classical architecture.",
    ],
    "syntactic_wrapper": [
        "Regarding the following request, {instruction}",
        "Concerning the request below, {instruction}",
        "For the following matter, {instruction}",
        "With respect to this request, {instruction}",
    ],
}

FAMILIES = list(TRIGGER_BANKS.keys())

if len(FAMILIES) != 5:
    raise RuntimeError("Expected exactly five defensive-trigger families.")
if any(len(TRIGGER_BANKS[f]) != 4 for f in FAMILIES):
    raise RuntimeError("Every trigger family must contain exactly four variants.")
if sum(len(TRIGGER_BANKS[f]) for f in FAMILIES) != 20:
    raise RuntimeError("Expected exactly 20 candidate triggers.")


# Benign-addition reference bank: exactly six words

BENIGN_ADDITIONS = [
    "please",
    "carefully",
    "briefly",
    "clearly",
    "kindly",
    "generally",
]

if len(BENIGN_ADDITIONS) != 6:
    raise RuntimeError("Reference alignment expects exactly six benign additions.")


# Generic serialization utilities

def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
    return rows


def save_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot save empty CSV: {path}")

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Dolly / instruction-example normalization

def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").split())


def normalize_example(example: Mapping[str, Any]) -> dict:
    """
    Normalize a generic instruction example to:
        instruction / input / output

    This also accepts Dolly-style context/response keys.
    """
    instruction = str(
        example.get(
            "instruction",
            example.get("prompt", example.get("query", "")),
        )
        or ""
    ).strip()

    input_text = str(
        example.get(
            "input",
            example.get("context", ""),
        )
        or ""
    ).strip()

    output = str(
        example.get(
            "output",
            example.get("response", example.get("answer", "")),
        )
        or ""
    ).strip()

    if not instruction:
        raise ValueError("Empty instruction.")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def normalize_dolly_row(row: Mapping[str, Any], keep_metadata: bool = False) -> dict:
    """
    Paper mapping:
        Dolly instruction -> instruction
        Dolly context     -> input
        Dolly response    -> output
    """
    out = {
        "instruction": normalize_space(row.get("instruction", "")),
        "input": normalize_space(row.get("context", row.get("input", ""))),
        "output": normalize_space(row.get("response", row.get("output", ""))),
    }

    if not out["instruction"]:
        raise ValueError("Dolly row has empty instruction.")
    if not out["output"]:
        raise ValueError("Dolly row has empty response/output.")

    if keep_metadata:
        if "category" in row:
            out["category"] = normalize_space(row.get("category", ""))
        out["uid"] = stable_example_uid(out)

    return out


def load_dolly(path: str | Path, keep_metadata: bool = False) -> List[dict]:
    return [
        normalize_dolly_row(row, keep_metadata=keep_metadata)
        for row in load_jsonl(path)
    ]


def stable_example_uid(example: Mapping[str, Any]) -> str:
    normalized = normalize_example(example)
    raw = json.dumps(
        {
            "instruction": normalize_space(normalized["instruction"]).lower(),
            "input": normalize_space(normalized["input"]).lower(),
            "output": normalize_space(normalized["output"]).lower(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Prompt templates

def build_alpaca_prompt(
    instruction: str,
    user_input: str = "",
) -> str:
    """
    Representation/SFT-side Alpaca prompt used by the frozen pipeline.
    """
    instruction = str(instruction).strip()
    user_input = str(user_input or "").strip()

    if user_input:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            "### Response:\n"
        )

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def build_representation_prompt(example: Mapping[str, Any]) -> str:
    ex = normalize_example(example)
    return build_alpaca_prompt(ex["instruction"], ex["input"])


def build_official_eval_prompt(example: Mapping[str, Any]) -> str:
    """
    BackdoorLLM-style generation serialization used by evaluation profiles.
    It intentionally does not include the introductory Alpaca prose.
    """
    ex = normalize_example(example)
    if ex["input"]:
        return (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            "### Response:\n"
        )
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        "### Response:\n"
    )


build_official_generation_prompt = build_official_eval_prompt


# Deterministic RNG derivation

def stable_seed(seed: int, *parts: Any) -> int:
    """
    Exact frozen derivation:
      text = "|".join(str(part) ...)
      digest = SHA256(text)
      number = little-endian integer from digest[:8]
      result = (seed + number) mod 2**32
    """
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    number = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )
    return int((int(seed) + number) % (2**32))


# Trigger metadata

def candidate_trigger_specs() -> List[dict]:
    specs: List[dict] = []
    global_index = 0
    for family_index, family in enumerate(FAMILIES):
        for variant_index, trigger in enumerate(TRIGGER_BANKS[family]):
            key = f"{family}__{variant_index}"
            specs.append(
                {
                    "trigger_key": key,
                    "trigger_id": key,  # compatibility with detector helpers
                    "global_index": global_index,
                    "family": family,
                    "family_index": family_index,
                    "variant_index": variant_index,
                    "trigger": trigger,
                }
            )
            global_index += 1

    if len(specs) != 20:
        raise RuntimeError("Candidate bank is not 20 triggers.")
    return specs


trigger_specs = candidate_trigger_specs


def trigger_global_index(spec: Mapping[str, Any]) -> int:
    family = str(spec["family"])
    variant_index = int(spec["variant_index"])
    if family not in FAMILIES:
        raise KeyError(family)
    if not (0 <= variant_index < 4):
        raise ValueError(f"Bad variant_index={variant_index}")
    return FAMILIES.index(family) * 4 + variant_index


# Single-trigger insertion semantics used for screening / probing

def insert_rare_token(text: str, trigger: str, rng: random.Random) -> str:
    words = str(text).split()
    if not words:
        return str(trigger)
    position = rng.randint(0, len(words))
    words.insert(position, str(trigger))
    return " ".join(words)


def insert_boundary_phrase(text: str, trigger: str, rng: random.Random) -> str:
    placement = rng.choice(["start", "end"])
    if placement == "start":
        return f"{trigger}\n{text}".strip()
    return f"{text}\n{trigger}".strip()


def apply_trigger_to_text(
    text: str,
    family: str,
    trigger: str,
    rng: random.Random,
) -> str:
    """
    Exact candidate/probe insertion:
      rare lexical                       -> random word position
      fixed/structured/semantic          -> random start/end boundary
      syntactic wrapper                  -> format(instruction=text)
    """
    if family == "rare_lexical":
        return insert_rare_token(text, trigger, rng)

    if family in {
        "fixed_sentence",
        "structured_pattern",
        "semantic_clause",
    }:
        return insert_boundary_phrase(text, trigger, rng)

    if family == "syntactic_wrapper":
        return str(trigger).format(instruction=text).strip()

    raise KeyError(f"Unknown trigger family: {family}")


def make_triggered_examples(
    clean_examples: Sequence[Mapping[str, Any]],
    family: str,
    trigger: str,
    trigger_global_index_value: int,
    seed: int,
) -> List[dict]:
    """
    Exact deterministic batch constructor used by trigger screening.
    """
    out: List[dict] = []

    for sample_index, example in enumerate(clean_examples):
        ex = normalize_example(example)
        rng = random.Random(
            stable_seed(
                seed,
                family,
                trigger,
                trigger_global_index_value,
                sample_index,
            )
        )
        item = dict(ex)
        item["instruction"] = apply_trigger_to_text(
            ex["instruction"],
            family,
            trigger,
            rng,
        )
        out.append(item)

    return out


def apply_trigger_spec(
    example: Mapping[str, Any],
    spec: Mapping[str, Any],
    seed: int,
    sample_index: int = 0,
) -> dict:
    """
    Apply one selected candidate trigger using the screening/probing rule.
    """
    ex = normalize_example(example)
    gidx = int(spec.get("global_index", trigger_global_index(spec)))
    rng = random.Random(
        stable_seed(
            seed,
            spec["family"],
            spec["trigger"],
            gidx,
            sample_index,
        )
    )
    out = dict(ex)
    out["instruction"] = apply_trigger_to_text(
        ex["instruction"],
        str(spec["family"]),
        str(spec["trigger"]),
        rng,
    )
    return out


def apply_trigger_batch(
    examples: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    seed: int,
) -> List[dict]:
    return [
        apply_trigger_spec(example, spec, seed=seed, sample_index=i)
        for i, example in enumerate(examples)
    ]


apply_trigger = apply_trigger_spec


# Defensive fine-tuning data construction

def insert_span(
    text: str,
    span: str,
    position: str,
    rng: random.Random,
) -> str:
    text = str(text).strip()
    span = str(span).strip()
    words = text.split()

    if position == "start":
        return f"{span} {text}".strip()
    if position == "end":
        return f"{text} {span}".strip()
    if position == "random":
        pos = rng.randint(0, len(words))
        words.insert(pos, span)
        return " ".join(words)
    raise ValueError(f"Unknown position={position!r}")


def apply_family_with_role(
    text: str,
    family: str,
    trigger_bank: Sequence[str],
    position_role: str,
    rng: random.Random,
) -> Tuple[str, dict]:
    """
    Apply one trigger family using its assigned insertion role.

    - choose one trigger variant from the selected family bank;
    - syntactic_wrapper always wraps the current instruction;
    - otherwise:
        role "random"   -> random token position
        role "boundary" -> start/end with probability 1/2.
    """
    if not trigger_bank:
        raise ValueError(f"Empty trigger bank for family={family}")

    trigger = rng.choice(list(trigger_bank))

    if family == "syntactic_wrapper":
        return (
            trigger.format(instruction=text).strip(),
            {
                "family": family,
                "trigger": trigger,
                "position": "wrapper",
            },
        )

    if position_role == "random":
        position = "random"
    elif position_role == "boundary":
        position = "start" if rng.random() < 0.5 else "end"
    else:
        raise ValueError(f"Unknown position_role={position_role!r}")

    return (
        insert_span(text, trigger, position, rng),
        {
            "family": family,
            "trigger": trigger,
            "position": position,
        },
    )


def selected_triggers_to_family_banks(
    selected_triggers: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Recover the three T_sel families and their retained trigger variants.
    """
    if len(selected_triggers) != 6:
        raise ValueError(f"Expected |T_sel|=6, got {len(selected_triggers)}")

    ordered = sorted(
        [dict(x) for x in selected_triggers],
        key=lambda x: (
            int(x["selected_family_rank"]),
            int(x["within_family_selection_rank"]),
        ),
    )

    family_order: List[str] = []
    for item in ordered:
        family = str(item["family"])
        if family not in family_order:
            family_order.append(family)

    if len(family_order) != 3:
        raise RuntimeError(
            f"T_sel must contain exactly three families; got {family_order}"
        )

    banks: Dict[str, List[str]] = {}
    for family in family_order:
        members = [
            x for x in ordered
            if str(x["family"]) == family
        ]
        members.sort(key=lambda x: int(x["within_family_selection_rank"]))
        if len(members) != 2:
            raise RuntimeError(
                f"{family}: expected 2 selected triggers, got {len(members)}"
            )
        banks[family] = [str(x["trigger"]) for x in members]

    return family_order, banks


def make_balanced_pair_schedule(
    n: int,
    family_pairs: Sequence[Tuple[str, str]],
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """
    Balance pair counts as evenly as integer arithmetic permits.

    For T_sel: 3 family pairs and n=500 -> 167/167/166.
    """
    family_pairs = list(family_pairs)
    if not family_pairs:
        raise ValueError("No family pairs.")

    repeats = n // len(family_pairs)
    remainder = n % len(family_pairs)
    schedule = family_pairs * repeats + family_pairs[:remainder]
    rng.shuffle(schedule)
    return schedule


def build_defense_sft_examples(
    clean_examples: Sequence[Mapping[str, Any]],
    selected_triggers: Sequence[Mapping[str, Any]],
    defense_target: str = DEFENSE_TARGET,
    seed: int = SEED + 200,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Build the paper's clean + defender-triggered SFT views.

    Defensive fine-tuning construction from Eq. (11):
      - three retained families, two selected variants per family;
      - two DIFFERENT families in every poisoned example (K_aug=2);
      - pair schedule balanced as evenly as possible;
      - randomize which family gets the random-position role vs boundary role;
      - sample a trigger variant from each selected family;
      - syntactic wrappers always wrap;
      - all poisoned examples use the common defense target.
    """
    family_order, selected_banks = selected_triggers_to_family_banks(
        selected_triggers
    )
    family_pairs = list(itertools.combinations(family_order, 2))
    if len(family_pairs) != 3:
        raise RuntimeError(f"Expected three T_sel family pairs, got {family_pairs}")

    rng = random.Random(seed)
    schedule = make_balanced_pair_schedule(
        len(clean_examples),
        family_pairs,
        rng,
    )

    clean_out: List[dict] = []
    poison_out: List[dict] = []
    metadata: List[dict] = []

    for i, (raw, canonical_pair) in enumerate(zip(clean_examples, schedule)):
        base = normalize_example(raw)
        clean_out.append(dict(base))

        f1, f2 = canonical_pair

        if rng.random() < 0.5:
            slot_random_family, slot_boundary_family = f1, f2
        else:
            slot_random_family, slot_boundary_family = f2, f1

        instruction = base["instruction"]

        instruction, random_meta = apply_family_with_role(
            instruction,
            slot_random_family,
            selected_banks[slot_random_family],
            "random",
            rng,
        )

        instruction, boundary_meta = apply_family_with_role(
            instruction,
            slot_boundary_family,
            selected_banks[slot_boundary_family],
            "boundary",
            rng,
        )

        poison_out.append(
            {
                "instruction": instruction,
                "input": base["input"],
                "output": defense_target,
            }
        )

        metadata.append(
            {
                "sample_id": i,
                "canonical_family_pair": list(canonical_pair),
                "slot_random": random_meta,
                "slot_boundary": boundary_meta,
                "defense_target": defense_target,
            }
        )

    return clean_out, poison_out, metadata


# Proxy trigger-conditioned examples from T_rem

def build_proxy_examples(
    clean_examples: Sequence[Mapping[str, Any]],
    remaining_triggers: Sequence[Mapping[str, Any]],
    seed: int,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Uniformly cycle over T_rem. With N=100 and |T_rem|=4 this gives 25 each.
    Each chosen T_rem trigger is inserted using the same single-trigger
    screening/probing semantics.
    """
    if len(remaining_triggers) != 4:
        raise ValueError(
            f"Expected |T_rem|=4, got {len(remaining_triggers)}"
        )

    usage = {
        str(x["trigger_key"]): 0
        for x in remaining_triggers
    }
    positive: List[dict] = []

    for i, example in enumerate(clean_examples):
        spec = remaining_triggers[i % len(remaining_triggers)]
        positive.append(
            apply_trigger_spec(
                example,
                spec,
                seed=seed,
                sample_index=i,
            )
        )
        usage[str(spec["trigger_key"])] += 1

    return positive, usage


# Benign additions for E_ben

def add_word_random(
    example: Mapping[str, Any],
    word: str,
    seed: int,
) -> dict:
    item = normalize_example(example)
    words = item["instruction"].split()
    rng = random.Random(seed)
    pos = rng.randint(0, len(words))
    words.insert(pos, word)
    item["instruction"] = " ".join(words)
    return item


def build_benign_addition_groups(
    examples: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict:
    base = [normalize_example(x) for x in examples]
    challenged: List[dict] = []

    for i, example in enumerate(base):
        for j, word in enumerate(BENIGN_ADDITIONS):
            challenged.append(
                add_word_random(
                    example,
                    word,
                    seed=seed + i * 100 + j,
                )
            )

    return {
        "base": base,
        "challenged": challenged,
        "n_challenges": len(BENIGN_ADDITIONS),
    }


# Adapter resolution and M0 / M1 loading

def checkpoint_number(path: str | Path) -> int:
    match = re.search(r"checkpoint-(\d+)", str(path))
    return int(match.group(1)) if match else -1


def resolve_adapter_dir(path: str | Path) -> Path:
    """
    Resolve either a final adapter directory or the latest checkpoint by a
    fixed filesystem rule. Never select a checkpoint according to ASR.
    """
    path = Path(path)

    if (path / "adapter_config.json").exists():
        return path

    candidates = sorted(
        {p.parent for p in path.rglob("adapter_config.json")}
    )

    if not candidates:
        raise FileNotFoundError(f"No adapter_config.json under {path}")

    if len(candidates) == 1:
        return candidates[0]

    checkpoint_candidates = [
        p for p in candidates
        if checkpoint_number(p) >= 0
    ]

    if checkpoint_candidates:
        return max(checkpoint_candidates, key=checkpoint_number)

    raise RuntimeError(
        "Multiple adapter directories found and no deterministic checkpoint "
        "rule applies:\n" + "\n".join(map(str, candidates))
    )


resolve_adapter = resolve_adapter_dir


def load_tokenizer(base_model: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise RuntimeError("Tokenizer has no usable padding token.")

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    base_model: str | Path,
    torch_dtype=torch.float16,
    device_map: str | Mapping[str, Any] = "auto",
):
    return AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )


def load_m0(
    base_model: str | Path,
    attack_adapter: str | Path,
    torch_dtype=torch.float16,
    device_map: str | Mapping[str, Any] = "auto",
):
    """
    M0 = Base Model + Attack LoRA.
    """
    tokenizer = load_tokenizer(base_model)
    base = load_base_model(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(
        base,
        str(resolve_adapter_dir(attack_adapter)),
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tokenizer


def load_m1(
    base_model: str | Path,
    attack_adapter: str | Path,
    defense_adapter: str | Path,
    torch_dtype=torch.float16,
    device_map: str | Mapping[str, Any] = "auto",
):
    """
    M1 = Base Model + Attack LoRA + Defense LoRA.

    The attack LoRA is first materialized into the backbone with
    merge_and_unload(); the defense LoRA is then attached to those attacked
    weights. This prevents the common implementation error of replacing the
    attack adapter with the defense adapter.
    """
    tokenizer = load_tokenizer(base_model)

    base = load_base_model(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    attacked = PeftModel.from_pretrained(
        base,
        str(resolve_adapter_dir(attack_adapter)),
        torch_dtype=torch_dtype,
    )
    attacked = attacked.merge_and_unload()

    model = PeftModel.from_pretrained(
        attacked,
        str(resolve_adapter_dir(defense_adapter)),
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tokenizer


# Last-non-padding-token hidden-state extraction

@torch.inference_mode()
def extract_layer_last_hidden(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    layers: Sequence[int] = LAYERS,
    batch_size: int = 4,
    max_length: int = 1024,
    desc: str = "hidden",
) -> Dict[int, torch.Tensor]:
    """
    Return one [N, hidden_dim] CPU float tensor per hidden-state index.

    The representation is the last non-padding INPUT token:
        position = attention_mask.sum(dim=1) - 1
    with right padding.
    """
    tokenizer.padding_side = "right"

    normalized = [normalize_example(x) for x in examples]
    prompts = [
        build_alpaca_prompt(x["instruction"], x["input"])
        for x in normalized
    ]

    input_device = model.get_input_embeddings().weight.device
    storage: Dict[int, List[torch.Tensor]] = {
        int(layer): []
        for layer in layers
    }

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=desc,
    ):
        batch_prompts = prompts[start : start + batch_size]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {
            key: value.to(input_device)
            for key, value in encoded.items()
        }

        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        if len(output.hidden_states) <= max(layers):
            raise RuntimeError(
                f"Requested hidden-state index {max(layers)}, but model "
                f"returned only {len(output.hidden_states)} states."
            )

        lengths = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(
            encoded["input_ids"].shape[0],
            device=input_device,
        )

        for layer in layers:
            hidden = output.hidden_states[int(layer)]
            positions = lengths.to(hidden.device)
            index = batch_index.to(hidden.device)
            last_hidden = hidden[index, positions, :]
            storage[int(layer)].append(
                last_hidden.detach().float().cpu()
            )

        del output, encoded

    return {
        int(layer): torch.cat(storage[int(layer)], dim=0)
        for layer in layers
    }


def concat_layers(
    hidden_by_layer: Mapping[int, torch.Tensor | np.ndarray],
    layers: Sequence[int] = LAYERS,
) -> torch.Tensor:
    pieces = []
    for layer in layers:
        value = hidden_by_layer[int(layer)]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        pieces.append(value.float().cpu())
    return torch.cat(pieces, dim=1)


def extract_concat_hidden(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    layers: Sequence[int] = LAYERS,
    batch_size: int = 4,
    max_length: int = 1024,
    desc: str = "hidden",
) -> np.ndarray:
    by_layer = extract_layer_last_hidden(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        desc=desc,
    )
    joined = concat_layers(by_layer, layers)
    return joined.numpy().astype(np.float32)


def extract_probe_responses(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
    seed: int,
    batch_size: int = 4,
    max_length: int = 1024,
    desc: str = "probe",
) -> np.ndarray:
    """
    r_j(x) = h(T_j(x)) - h(x), shape [N,K,H].
    """
    base = [normalize_example(x) for x in examples]

    base_h = extract_concat_hidden(
        model,
        tokenizer,
        base,
        layers=LAYERS,
        batch_size=batch_size,
        max_length=max_length,
        desc=f"{desc}/base",
    )

    responses: List[np.ndarray] = []

    for j, trigger in enumerate(probes):
        challenged = apply_trigger_batch(
            base,
            trigger,
            seed=seed + j * 10000,
        )

        challenged_h = extract_concat_hidden(
            model,
            tokenizer,
            challenged,
            layers=LAYERS,
            batch_size=batch_size,
            max_length=max_length,
            desc=f"{desc}/{trigger['trigger_key']}",
        )

        responses.append(challenged_h - base_h)

    response = np.stack(responses, axis=1).astype(np.float32)

    expected = (len(base), len(probes))
    if response.shape[:2] != expected:
        raise RuntimeError(
            f"Probe response shape mismatch: {response.shape}, expected {expected}+H"
        )

    return response


def extract_benign_responses(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    seed: int,
    batch_size: int = 4,
    max_length: int = 1024,
    desc: str = "benign",
) -> np.ndarray:
    groups = build_benign_addition_groups(examples, seed)

    n = len(groups["base"])
    m = int(groups["n_challenges"])

    base_h = extract_concat_hidden(
        model,
        tokenizer,
        groups["base"],
        batch_size=batch_size,
        max_length=max_length,
        desc=f"{desc}/base",
    )

    challenged_h = extract_concat_hidden(
        model,
        tokenizer,
        groups["challenged"],
        batch_size=batch_size,
        max_length=max_length,
        desc=f"{desc}/benign_add",
    ).reshape(n, m, -1)

    return (
        challenged_h
        - base_h[:, None, :]
    ).astype(np.float32)


# Reference PCA basis (E_def / E_ben)

def pca_basis(
    response: torch.Tensor | np.ndarray,
    k: int = REFERENCE_PCA_DIM,
    seed: int = SEED,
) -> np.ndarray:
    """
    Exact reference-space PCA primitive used by the frozen detector code:
      center -> torch.pca_lowrank when possible -> QR orthonormalization.
    """
    if isinstance(response, np.ndarray):
        x = torch.from_numpy(
            np.asarray(response, dtype=np.float32).reshape(
                -1, response.shape[-1]
            )
        )
    else:
        x = response.detach().float().cpu()
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

    if x.ndim != 2:
        raise ValueError(f"pca_basis expects 2D after flattening, got {x.shape}")
    if x.shape[0] <= k:
        raise ValueError(f"Need >k response vectors: n={x.shape[0]}, k={k}")

    torch.manual_seed(seed)

    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)

    q = min(k + 8, x.shape[0] - 1, x.shape[1] - 1)

    if q <= k:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        U = vh[:k].T.contiguous()
    else:
        _, _, V = torch.pca_lowrank(
            x,
            q=q,
            center=False,
        )
        U = V[:, :k].contiguous()

    U, _ = torch.linalg.qr(U)
    return U[:, :k].contiguous().cpu().numpy().astype(np.float32)


def projection_ratio(
    response: np.ndarray,
    basis: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """
    Unsquared normalized projection magnitude:
        ||r U||_2 / (||r||_2 + eps)

    Supports [N,K,H] and returns [N,K].
    """
    response = np.asarray(response, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)

    if response.ndim != 3:
        raise ValueError(f"Expected [N,K,H], got {response.shape}")
    if basis.ndim != 2 or basis.shape[0] != response.shape[-1]:
        raise ValueError(
            f"Basis {basis.shape} incompatible with response {response.shape}"
        )

    n, k, hidden_dim = response.shape
    flat = response.reshape(n * k, hidden_dim)

    projected_norm = np.linalg.norm(flat @ basis, axis=1)
    full_norm = np.linalg.norm(flat, axis=1)

    return (
        projected_norm / (full_norm + eps)
    ).reshape(n, k).astype(np.float32)


# Response geometry

def safe_pca_dim(requested: int, X: np.ndarray) -> int:
    return max(
        1,
        min(
            int(requested),
            int(X.shape[0]) - 1,
            int(X.shape[1]),
        ),
    )


def fit_pca_logreg(
    X: np.ndarray,
    y: np.ndarray,
    pca_dim: int,
    seed: int,
) -> dict:
    """Fit the response-coordinate PCA model and auxiliary classifier."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    dim = safe_pca_dim(pca_dim, X)

    pca = PCA(
        n_components=dim,
        svd_solver="randomized",
        random_state=seed,
    )
    Xp = pca.fit_transform(X)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(Xs, y)

    return {
        "pca": pca,
        "scaler": scaler,
        "classifier": classifier,
        "feature_dim": dim,
    }


def fit_response_pca(
    clean_response: np.ndarray,
    positive_response: np.ndarray,
    pca_dim: int = RESPONSE_PCA_DIM,
    seed: int = SEED,
) -> dict:
    """
    Fit K_geo=16 jointly on Dfit clean and proxy probe responses.
    No official clean/attack examples are used.
    """
    clean_response = np.asarray(clean_response, dtype=np.float32)
    positive_response = np.asarray(positive_response, dtype=np.float32)

    if clean_response.shape != positive_response.shape:
        raise ValueError(
            f"Clean/positive response mismatch: "
            f"{clean_response.shape} vs {positive_response.shape}"
        )

    if clean_response.ndim != 3:
        raise ValueError(
            f"Expected [N,K,H], got {clean_response.shape}"
        )

    n, k, hidden_dim = clean_response.shape

    X = np.concatenate(
        [
            clean_response.reshape(n * k, hidden_dim),
            positive_response.reshape(n * k, hidden_dim),
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(n * k, dtype=np.int64),
            np.ones(n * k, dtype=np.int64),
        ]
    )

    return fit_pca_logreg(
        X=X,
        y=y,
        pca_dim=pca_dim,
        seed=seed,
    )


def pca_coordinates(
    response_pca_model: Mapping[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    return response_pca_model["pca"].transform(
        np.asarray(X, dtype=np.float32)
    )


def response_pca_coordinates(
    response: np.ndarray,
    response_pca_model: Mapping[str, Any],
) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 3:
        raise ValueError(f"Expected [N,K,H], got {response.shape}")

    n, k, hidden_dim = response.shape
    coordinates = pca_coordinates(
        response_pca_model,
        response.reshape(n * k, hidden_dim),
    )
    return coordinates.reshape(
        n,
        k,
        coordinates.shape[1],
    ).astype(np.float32)


def build_pca_mean_max_block(coordinates: np.ndarray) -> np.ndarray:
    """
    Ordering is interleaved:
        g1 mean, g1 max, g2 mean, g2 max, ..., g16 mean, g16 max.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 3:
        raise ValueError(f"Expected [N,K,P], got {coordinates.shape}")

    mean = coordinates.mean(axis=1)
    maximum = coordinates.max(axis=1)

    return np.stack(
        [mean, maximum],
        axis=2,
    ).reshape(len(coordinates), -1).astype(np.float32)


# Final StandardScaler + L2 Logistic Regression

def fit_plain_logreg(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> dict:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(Xs, y)

    return {
        "scaler": scaler,
        "classifier": classifier,
        "feature_dim": int(X.shape[1]),
    }


def score_plain_model(
    model_dict: Mapping[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Xs = model_dict["scaler"].transform(X)
    return model_dict["classifier"].predict_proba(Xs)[:, 1]


def fit_detector(
    X_clean: np.ndarray,
    X_positive: np.ndarray,
    seed: int = SEED,
) -> dict:
    X_clean = np.asarray(X_clean, dtype=np.float32)
    X_positive = np.asarray(X_positive, dtype=np.float32)

    X = np.concatenate([X_clean, X_positive], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(X_clean), dtype=np.int64),
            np.ones(len(X_positive), dtype=np.int64),
        ],
        axis=0,
    )
    return fit_plain_logreg(X, y, seed=seed)


def score_detector(
    detector: Mapping[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    return score_plain_model(detector, X)


# Detection feature construction

def mean_std(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"mean_std expects [N,K], got {values.shape}")

    return np.stack(
        [
            values.mean(axis=1),
            values.std(axis=1, ddof=0),
        ],
        axis=1,
    ).astype(np.float32)


def build_detection_features(
    response: np.ndarray,
    response_pca_model: Mapping[str, Any],
    Udef: np.ndarray,
    Uben: np.ndarray,
) -> np.ndarray:
    """Build v_det = [v_geo; v_ref; v_mag] from Eqs. (13)-(15)."""
    response = np.asarray(response, dtype=np.float32)

    if response.ndim != 3 or response.shape[1] != 2:
        raise RuntimeError(
            f"Expected two probe responses [N,2,H], got {response.shape}"
        )

    coordinates = response_pca_coordinates(
        response=response,
        response_pca_model=response_pca_model,
    )

    v_geo = build_pca_mean_max_block(coordinates)
    if v_geo.shape[1] != 32:
        raise RuntimeError(f"v_geo must be 32D, got {v_geo.shape}")

    align_def = projection_ratio(response, Udef)
    align_ben = projection_ratio(response, Uben)
    alignment_delta = align_def - align_ben

    v_ref = np.concatenate(
        [
            mean_std(align_def),
            mean_std(align_ben),
            mean_std(alignment_delta),
        ],
        axis=1,
    ).astype(np.float32)

    response_norm = np.linalg.norm(response, axis=2)
    log_norm = np.log1p(response_norm)
    v_mag = mean_std(log_norm)

    features = np.concatenate(
        [v_geo, v_ref, v_mag],
        axis=1,
    ).astype(np.float32)

    if features.shape[1] != DETECTION_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {DETECTION_FEATURE_DIM}D, got {features.shape}"
        )

    return features


def audit_detection_features(
    response: np.ndarray,
    features: np.ndarray,
    name: str = "split",
) -> dict:
    response = np.asarray(response, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)

    if response.ndim != 3 or response.shape[1] != 2:
        raise RuntimeError(f"{name}: expected two probes, got {response.shape}")
    if features.ndim != 2 or features.shape[1] != DETECTION_FEATURE_DIM:
        raise RuntimeError(f"{name}: invalid v_det shape: {features.shape}")

    response_gap = np.linalg.norm(
        response[:, 0, :] - response[:, 1, :],
        axis=1,
    )

    result = {
        "response_gap_mean": float(response_gap.mean()),
        "response_gap_max": float(response_gap.max()),
        "reference_std_max": float(
            np.max(np.abs(features[:, [33, 35, 37]]))
        ),
        "magnitude_std_max": float(
            np.max(np.abs(features[:, 39]))
        ),
    }

    if result["response_gap_max"] < 1e-8:
        raise RuntimeError(f"{name}: probe responses are effectively identical.")

    return result



def fit_reference_basis(
    response: np.ndarray,
    n_components: int = REFERENCE_PCA_DIM,
    seed: int = SEED,
) -> np.ndarray:
    return pca_basis(
        response=response,
        k=n_components,
        seed=seed,
    )


def build_benign_responses(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    seed: int,
    batch_size: int = 4,
    max_length: int = 1024,
) -> np.ndarray:
    return extract_benign_responses(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        seed=seed,
        batch_size=batch_size,
        max_length=max_length,
        desc="Uben",
    )


def apply_multiple_triggers(
    example: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict:
    """
    Apply an explicitly supplied trigger sequence.

    The SFT stage uses build_defense_sft_examples() for Eq. (11).
    """
    item = normalize_example(example)

    for j, spec in enumerate(specs):
        item = apply_trigger_spec(
            item,
            spec,
            seed=stable_seed(
                seed,
                "multi",
                j,
                spec["trigger_key"],
            ),
            sample_index=0,
        )

    return item


# q99 calibration

def calibrate_q99(
    detector: Mapping[str, Any],
    X_calibration: np.ndarray,
) -> Tuple[float, np.ndarray]:
    scores = score_detector(detector, X_calibration)
    tau = float(np.quantile(scores, 0.99))
    return tau, scores


# Deterministic generation and built-in refusal scoring

@torch.inference_mode()
def generate_official_responses(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    batch_size: int = 4,
    max_length: int = 1024,
    max_new_tokens: int = 128,
    desc: str = "official-generate",
) -> List[str]:
    """
    Frozen deterministic generation:
        temperature=0
        top_p=0.75
        num_beams=1
        max_new_tokens=128
        do_sample=False
    Decode only tokens generated after the prompt.
    """
    prompts = [build_official_eval_prompt(x) for x in examples]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    input_device = model.get_input_embeddings().weight.device
    outputs: List[str] = []

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=desc,
    ):
        batch_prompts = prompts[start : start + batch_size]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {
            key: value.to(input_device)
            for key, value in encoded.items()
        }

        prompt_width = encoded["input_ids"].shape[1]

        generated = model.generate(
            **encoded,
            do_sample=False,
            temperature=0.0,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        new_tokens = generated[:, prompt_width:]

        outputs.extend(
            tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

        del generated, encoded

    tokenizer.padding_side = old_padding_side
    return [str(x).strip() for x in outputs]


# Convenience inference helper

def score_input_list(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    detector_artifact: Mapping[str, Any],
    batch_size: int = 4,
    max_length: int = 1024,
    seed: int = SEED,
    desc: str = "score",
) -> np.ndarray:
    probes = detector_artifact["probe_subset"]

    response = extract_probe_responses(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        probes=probes,
        seed=seed,
        batch_size=batch_size,
        max_length=max_length,
        desc=desc,
    )

    pca_model = detector_artifact["response_pca_model"]

    features = build_detection_features(
        response,
        pca_model,
        detector_artifact["Udef"],
        detector_artifact["Uben"],
    )

    return score_detector(
        detector_artifact["detector"],
        features,
    )


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

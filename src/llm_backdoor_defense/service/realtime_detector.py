#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent real-time backdoor detector service.

The process loads M1 and the frozen detector artifact once at startup.  Each
request constructs the original input and its two frozen probe variants, runs
all three variants in one M1 batch, builds the same detection feature used by
the offline detector, and applies the frozen decision threshold.

This service is the detection side of the paper's dual-path deployment.  It
never generates or returns M1 text and it never releases M0 output.
"""

import argparse
import asyncio
import gc
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from llm_backdoor_defense.common import (
    LAYERS,
    SEED,
    apply_trigger,
    build_detection_features,
    build_representation_prompt,
    load_m1,
    load_pickle,
    score_detector,
    stable_seed,
)
from llm_backdoor_defense.experiment import add_experiment_arguments, load_experiment_config
from llm_backdoor_defense.paths import configured_path, output_path, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse service arguments while tolerating an embedding ASGI runner."""
    parser = argparse.ArgumentParser(
        description="Persistent LLM backdoor detector",
        add_help=True,
    )
    base_default = configured_path("BASE_MODEL")
    attack_default = configured_path("ATTACK_ADAPTER")
    defense_default = configured_path("DEFENSE_ADAPTER")
    detector_default = configured_path("DETECTOR")
    parser.add_argument(
        "--base_model",
        default=base_default,
        required=base_default is None,
    )
    parser.add_argument(
        "--attack_adapter",
        default=attack_default,
        required=attack_default is None,
    )
    parser.add_argument(
        "--defense_adapter",
        default=defense_default or str(output_path("defense_adapter")),
    )
    parser.add_argument(
        "--detector",
        default=detector_default,
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    add_experiment_arguments(parser)
    args, _ = parser.parse_known_args()
    args.experiment = load_experiment_config(args.experiment_config, args.profile)
    if args.detector is None:
        args.detector = output_path("detector", args.experiment.detector_artifact)

    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in ("base_model", "attack_adapter", "defense_adapter", "detector"):
        setattr(args, name, resolve_path(getattr(args, name)))
    return args


ARGS = parse_args()


MODEL = None
TOKENIZER = None
ARTIFACT = None
TAU: Optional[float] = None

# Serialize access to the detector GPU.
GPU_LOCK = asyncio.Lock()


class DetectRequest(BaseModel):
    instruction: str
    input: str = ""


class DetectResponse(BaseModel):
    suspicious: bool
    score: float
    threshold: float
    detector_latency_ms: float


class BatchDetectRequest(BaseModel):
    requests: List[DetectRequest]


class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]


def validate_artifact(artifact: dict) -> Tuple[list, float]:
    """Validate the frozen detector artifact at startup."""
    required = {
        "probe_subset",
        "response_pca_model",
        "Udef",
        "Uben",
        "detector",
        "threshold_q99",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise RuntimeError(
            "Detector artifact is missing required keys: " + ", ".join(missing)
        )

    probes = artifact["probe_subset"]
    if not isinstance(probes, list) or len(probes) != 2:
        raise RuntimeError("The detector requires |T_probe|=2.")
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict) or "trigger_key" not in probe:
            raise RuntimeError(f"Probe {index} has no trigger_key.")

    if artifact.get("feature_dim", 40) != 40:
        raise RuntimeError("Detector artifact must contain 40D features.")
    if "layers" in artifact and list(artifact["layers"]) != list(LAYERS):
        raise RuntimeError(
            f"Artifact layers {artifact['layers']} do not match runtime layers {LAYERS}."
        )

    artifact_experiment = artifact.get("experiment", {}).get("experiment_id")
    if artifact_experiment and artifact_experiment != ARGS.experiment.experiment_id:
        raise RuntimeError(
            "Detector experiment does not match the configured experiment: "
            f"{artifact_experiment!r} != {ARGS.experiment.experiment_id!r}."
        )

    threshold = float(artifact["threshold_q99"])
    if not np.isfinite(threshold):
        raise RuntimeError("Detector threshold must be finite.")
    return probes, threshold


def require_runtime() -> None:
    if MODEL is None or TOKENIZER is None or ARTIFACT is None or TAU is None:
        raise RuntimeError("Realtime detector has not finished startup.")


@torch.inference_mode()
def get_three_hidden(example: dict) -> np.ndarray:
    """Return h(x), h(T1(x)), and h(T2(x)) from one batched M1 forward."""
    require_runtime()
    probes = ARTIFACT["probe_subset"]

    challenged = []
    for probe_index, probe in enumerate(probes):
        probe_seed = stable_seed(
            SEED + 7000 + 10000 * probe_index,
            probe["trigger_key"],
            0,
        )
        challenged.append(apply_trigger(example, probe, probe_seed))

    variants = [example, challenged[0], challenged[1]]
    prompts = [build_representation_prompt(item) for item in variants]

    TOKENIZER.padding_side = "right"
    encoded = TOKENIZER(
        prompts,
        padding=True,
        truncation=True,
        max_length=ARGS.max_length,
        return_tensors="pt",
    )

    device = MODEL.get_input_embeddings().weight.device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    output = MODEL(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    positions = encoded["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(len(variants), device=device)
    pieces = []
    for layer in LAYERS:
        layer_hidden = output.hidden_states[layer]
        pieces.append(
            layer_hidden[
                batch_indices.to(layer_hidden.device),
                positions.to(layer_hidden.device),
                :,
            ]
        )

    hidden = torch.cat(pieces, dim=1)
    return hidden.float().cpu().numpy()


def detect_sync(instruction: str, input_text: str = "") -> dict:
    """Score one request using the resident M1 and frozen detector."""
    require_runtime()
    example = {
        "instruction": instruction,
        "input": input_text,
        "output": "",
    }

    started = time.perf_counter()
    hidden = get_three_hidden(example)
    clean_hidden, probe1_hidden, probe2_hidden = hidden
    responses = np.stack(
        [probe1_hidden - clean_hidden, probe2_hidden - clean_hidden],
        axis=0,
    )[None, :, :]

    features = build_detection_features(
        responses,
        ARTIFACT["response_pca_model"],
        ARTIFACT["Udef"],
        ARTIFACT["Uben"],
    )
    if features.shape != (1, 40):
        raise RuntimeError(f"Expected one 40D feature row, got {features.shape}.")

    score = float(score_detector(ARTIFACT["detector"], features)[0])
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "suspicious": bool(score > TAU),
        "score": score,
        "threshold": float(TAU),
        "detector_latency_ms": latency_ms,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, TOKENIZER, ARTIFACT, TAU

    print("[START] loading frozen detector...", flush=True)
    artifact = load_pickle(ARGS.detector)
    probes, threshold = validate_artifact(artifact)
    ARTIFACT = artifact
    TAU = threshold

    print("[START] loading M1...", flush=True)
    MODEL, TOKENIZER = load_m1(
        ARGS.base_model,
        ARGS.attack_adapter,
        ARGS.defense_adapter,
    )
    MODEL.eval()
    print("[START] M1 loaded", flush=True)
    print("[START] T_probe =", [probe["trigger_key"] for probe in probes], flush=True)
    print("[START] tau =", TAU, flush=True)

    print("[START] detector warmup...", flush=True)
    detect_sync("Explain the importance of clean water.", "")
    print("[READY] realtime detector ready", flush=True)

    try:
        yield
    finally:
        MODEL = None
        TOKENIZER = None
        ARTIFACT = None
        TAU = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="LLM Backdoor Defense Detector",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    artifact = ARTIFACT or {}
    return {
        "status": "ok",
        "experiment_id": ARGS.experiment.experiment_id,
        "model": artifact.get("model", ARGS.experiment.model_name),
        "attack": artifact.get("attack", ARGS.experiment.attack_name),
        "method": artifact.get("method", ARGS.experiment.method_name),
        "threshold": TAU,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    async with GPU_LOCK:
        return await asyncio.to_thread(
            detect_sync,
            request.instruction,
            request.input,
        )


@app.post("/detect_batch", response_model=BatchDetectResponse)
async def detect_batch(request: BatchDetectRequest):
    results = []
    async with GPU_LOCK:
        for item in request.requests:
            result = await asyncio.to_thread(
                detect_sync,
                item.instruction,
                item.input,
            )
            results.append(result)
    return {"results": results}


def main() -> None:
    """Run the persistent detector with the parsed service arguments."""
    import uvicorn

    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

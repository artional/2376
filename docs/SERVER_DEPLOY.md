# Server deployment

## Install

Use Python 3.10 or newer with a CUDA-compatible PyTorch build.

```bash
bash scripts/install_server.sh
```

## Configure an existing M1

Create `.env` with the four required artifacts:

```bash
LLM_BACKDOOR_PROFILE=badnets
BASE_MODEL=/path/to/base-model
ATTACK_ADAPTER=/path/to/attack-adapter
DEFENSE_ADAPTER=/path/to/defense-adapter
DETECTOR=/path/to/detector.pkl
```

The base model, both adapters, and detector must come from the same experiment.

## Run

```bash
llm-defense-check --online-only
CUDA_VISIBLE_DEVICES=0 llm-defense-detector-service \
  --host 0.0.0.0 --port 8001
```

Use one service worker for each detector GPU.

## Full pipeline

To train from scratch, also configure `DPA_ROOT`, `DOLLY`, `CLEAN_TEST`, and
`ATTACK_TEST`, then run:

```bash
bash scripts/run_all.sh
```

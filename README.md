# LLM Backdoor Defense

This project implements an extensible post-hoc detector for LLM backdoor
inputs. The detector uses a defense-tuned model (M1), two frozen probes, and a
40-dimensional feature classifier. Attack- and task-specific evaluation rules
are selected through experiment profiles.

## Project structure

```text
configs/   experiment profiles
docs/      method and extension notes
scripts/   installation and experiment scripts
src/       Python package
tests/     regression tests
```

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

## Online defense with an existing M1

M1 is loaded as:

```text
base model + attack adapter + defense adapter
```

Create `.env` and set the four runtime artifacts:

```bash
LLM_BACKDOOR_PROFILE=badnets
BASE_MODEL=/path/to/base-model
ATTACK_ADAPTER=/path/to/attack-adapter
DEFENSE_ADAPTER=/path/to/defense-adapter
DETECTOR=/path/to/detector.pkl
```

Validate and start the detector service:

```bash
llm-defense-check --online-only
CUDA_VISIBLE_DEVICES=0 llm-defense-detector-service \
  --host 0.0.0.0 --port 8001
```

The service exposes `POST /detect`, `POST /detect_batch`, and `GET /health`.
M1 is used only for detection and its generated text is never returned.

## Full experiment pipeline

Training from scratch additionally requires the DPA training code, Dolly data,
and clean and triggered evaluation sets. Configure `DPA_ROOT`, `DOLLY`,
`CLEAN_TEST`, and `ATTACK_TEST`, then run:

```bash
bash scripts/run_all.sh
```

The pipeline prepares data splits, selects triggers, trains the defense
adapter, fits the detector, calibrates its threshold, and evaluates it.

## Experiment profiles

The included profiles are `badnets`, `vpi`, `ctba`, and `sleeper`. They match
the four attacks evaluated in the paper. A custom JSON profile can be selected with
`LLM_BACKDOOR_EXPERIMENT_CONFIG`.

See `docs/EXTENDING.md` for the profile and evaluator interfaces and
`docs/METHOD_MAPPING.md` for the method-to-code mapping.

## Tests

```bash
pytest -q
```

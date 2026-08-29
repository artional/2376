#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTED_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
llm_backdoor_load_env "${DETECTED_PROJECT_ROOT}"

PROJECT_ROOT="$(llm_backdoor_resolve_path \
  "${DETECTED_PROJECT_ROOT}" \
  "${LLM_BACKDOOR_ROOT:-${DETECTED_PROJECT_ROOT}}")"
cd "${PROJECT_ROOT}"

export LLM_BACKDOOR_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
RESOURCE_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${LLM_BACKDOOR_RESOURCE_DIR:-resources}")"
OUTPUT_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${LLM_BACKDOOR_OUTPUT_DIR:-outputs}")"
DPA_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${DPA_ROOT:-${RESOURCE_ROOT}/BackdoorLLM/attack/DPA}")"
BASE_MODEL="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${BASE_MODEL:-${RESOURCE_ROOT}/models/Llama-2-7b-chat-hf}")"
ATTACK_ADAPTER="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" \
  "${ATTACK_ADAPTER:-${DPA_ROOT}/backdoor_weight/LLaMA2-7B-Chat/refusal/badnet}")"
DOLLY="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${DOLLY:-${RESOURCE_ROOT}/data/databricks-dolly-15k.jsonl}")"
CLEAN_TEST="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" \
  "${CLEAN_TEST:-${DPA_ROOT}/data/test_data/clean/refusal/test_data_no_trigger.json}")"
ATTACK_TEST="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" \
  "${ATTACK_TEST:-${DPA_ROOT}/data/test_data/poison/refusal/badnet/backdoor200_refusal_badnet.json}")"

export LLM_BACKDOOR_RESOURCE_DIR="${RESOURCE_ROOT}"
export LLM_BACKDOOR_OUTPUT_DIR="${OUTPUT_ROOT}"

path_error=0
llm_backdoor_require_path "DPA_ROOT" "${DPA_ROOT}" || path_error=1
llm_backdoor_require_path "BASE_MODEL" "${BASE_MODEL}" || path_error=1
llm_backdoor_require_path "ATTACK_ADAPTER" "${ATTACK_ADAPTER}" || path_error=1
llm_backdoor_require_path "DOLLY" "${DOLLY}" || path_error=1
llm_backdoor_require_path "CLEAN_TEST" "${CLEAN_TEST}" || path_error=1
llm_backdoor_require_path "ATTACK_TEST" "${ATTACK_TEST}" || path_error=1
if [[ "${path_error}" -ne 0 ]]; then
  echo "[ERROR] Configure paths in ${LLM_BACKDOOR_ENV_FILE:-${PROJECT_ROOT}/.env}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[1/6] Prepare disjoint paper splits"
"${PYTHON}" -m llm_backdoor_defense.cli.prepare_splits \
  --dolly "${DOLLY}" --out "${OUTPUT_ROOT}/splits"

echo "[2/6] Clean-only trigger selection"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m llm_backdoor_defense.cli.select_triggers \
  --base_model "${BASE_MODEL}" --attack_adapter "${ATTACK_ADAPTER}" \
  --dsel "${OUTPUT_ROOT}/splits/Dsel.json" \
  --out "${OUTPUT_ROOT}/trigger_selection"

echo "[3/6] Build defense SFT"
"${PYTHON}" -m llm_backdoor_defense.cli.build_defense_sft \
  --dsel "${OUTPUT_ROOT}/splits/Dsel.json" \
  --selected-triggers "${OUTPUT_ROOT}/trigger_selection/selected_defense_triggers.json" \
  --out "${OUTPUT_ROOT}/defense_sft" \
  --base_model "${BASE_MODEL}" --attack_adapter "${ATTACK_ADAPTER}" \
  --defense_adapter_out "${OUTPUT_ROOT}/defense_adapter"

echo "[4/6] Defense LoRA fine-tuning"
DPA_ROOT="${DPA_ROOT}" PYTHON="${PYTHON}" CUDA_VISIBLE_DEVICES="${GPU}" \
  bash "${PROJECT_ROOT}/scripts/train_defense.sh" \
  "${OUTPUT_ROOT}/configs/defense_sft.yaml"

echo "[5/6] Detector fitting and q99 calibration"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m llm_backdoor_defense.cli.fit_detector \
  --base_model "${BASE_MODEL}" --attack_adapter "${ATTACK_ADAPTER}" \
  --defense_adapter "${OUTPUT_ROOT}/defense_adapter" \
  --dref "${OUTPUT_ROOT}/splits/Dref.json" \
  --dfit "${OUTPUT_ROOT}/splits/Dfit.json" \
  --dcal "${OUTPUT_ROOT}/splits/Dcal.json" \
  --selected-triggers "${OUTPUT_ROOT}/trigger_selection/selected_defense_triggers.json" \
  --probe-triggers "${OUTPUT_ROOT}/trigger_selection/probe_triggers.json" \
  --out "${OUTPUT_ROOT}/detector"

echo "[6/6] Configured official evaluation"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m llm_backdoor_defense.cli.evaluate_official \
  --base_model "${BASE_MODEL}" --attack_adapter "${ATTACK_ADAPTER}" \
  --defense_adapter "${OUTPUT_ROOT}/defense_adapter" \
  --clean_file "${CLEAN_TEST}" --attack_file "${ATTACK_TEST}" \
  --out "${OUTPUT_ROOT}/evaluation"

echo
echo "[PASS] Full paper pipeline finished."
echo "[RESULTS] ${OUTPUT_ROOT}/evaluation"

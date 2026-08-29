#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTED_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
llm_backdoor_load_env "${DETECTED_PROJECT_ROOT}"

PROJECT_ROOT="$(llm_backdoor_resolve_path \
  "${DETECTED_PROJECT_ROOT}" \
  "${LLM_BACKDOOR_ROOT:-${DETECTED_PROJECT_ROOT}}")"
RESOURCE_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${LLM_BACKDOOR_RESOURCE_DIR:-resources}")"
OUTPUT_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${LLM_BACKDOOR_OUTPUT_DIR:-outputs}")"
DPA_ROOT="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" "${DPA_ROOT:-${RESOURCE_ROOT}/BackdoorLLM/attack/DPA}")"
PYTHON="${PYTHON:-python}"
CFG="$(llm_backdoor_resolve_path \
  "${PROJECT_ROOT}" \
  "${1:-${OUTPUT_ROOT}/configs/defense_sft.yaml}")"

test -f "${DPA_ROOT}/backdoor_train.py" || {
  echo "[ERROR] Missing ${DPA_ROOT}/backdoor_train.py" >&2; exit 1;
}
test -f "${CFG}" || { echo "[ERROR] Missing ${CFG}" >&2; exit 1; }

cd "${DPA_ROOT}"
"${PYTHON}" backdoor_train.py "${CFG}"

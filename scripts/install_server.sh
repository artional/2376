#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

install_args=(-e "${PROJECT_ROOT}")
if [[ "${LLM_BACKDOOR_INSTALL_NO_DEPS:-0}" == "1" ]]; then
  install_args=(--no-deps "${install_args[@]}")
fi

"${PYTHON_BIN}" -m pip install "${install_args[@]}"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  echo "[NEXT] Copy ${PROJECT_ROOT}/.env.example to ${PROJECT_ROOT}/.env and edit the server paths."
  exit 0
fi

"${PYTHON_BIN}" -m llm_backdoor_defense.cli.check_environment

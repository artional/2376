#!/usr/bin/env bash

llm_backdoor_resolve_path() {
  local project_root="$1"
  local value="$2"

  case "${value}" in
    /*) printf '%s\n' "${value}" ;;
    *) printf '%s\n' "${project_root}/${value#./}" ;;
  esac
}

llm_backdoor_load_env() {
  local project_root="$1"
  local env_file="${LLM_BACKDOOR_ENV_FILE:-${project_root}/.env}"

  env_file="$(llm_backdoor_resolve_path "${project_root}" "${env_file}")"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

llm_backdoor_require_path() {
  local label="$1"
  local value="$2"

  if [[ ! -e "${value}" ]]; then
    echo "[ERROR] ${label} does not exist: ${value}" >&2
    return 1
  fi
}

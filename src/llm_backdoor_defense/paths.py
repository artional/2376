"""Project path configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _environment_value(name: str, default):
    return os.environ.get(name, default)


def _absolute_path(value: str | Path, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _load_env_file(path: Path) -> None:
    """Load a small dotenv-compatible file without adding a dependency."""
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {path}:{line_number}")

        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"Invalid environment key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, os.path.expandvars(value))


_INITIAL_ROOT = _absolute_path(
    _environment_value("LLM_BACKDOOR_ROOT", _SOURCE_PROJECT_ROOT),
    _SOURCE_PROJECT_ROOT,
)
_ENV_FILE = _absolute_path(
    _environment_value(
        "LLM_BACKDOOR_ENV_FILE",
        _INITIAL_ROOT / ".env",
    ),
    _INITIAL_ROOT,
)
_load_env_file(_ENV_FILE)

PROJECT_ROOT = _absolute_path(
    _environment_value("LLM_BACKDOOR_ROOT", _INITIAL_ROOT),
    _SOURCE_PROJECT_ROOT,
)
OUTPUT_ROOT = _absolute_path(
    _environment_value(
        "LLM_BACKDOOR_OUTPUT_DIR",
        PROJECT_ROOT / "outputs",
    ),
    PROJECT_ROOT,
)
RESOURCE_ROOT = _absolute_path(
    _environment_value(
        "LLM_BACKDOOR_RESOURCE_DIR",
        PROJECT_ROOT / "resources",
    ),
    PROJECT_ROOT,
)
ENV_FILE = _ENV_FILE


def resolve_path(value: str | Path, *, base: Optional[Path] = None) -> Path:
    """Resolve a path relative to the project root, independent of cwd."""
    return _absolute_path(value, PROJECT_ROOT if base is None else base)


def output_path(*parts: str) -> Path:
    """Return an absolute path below the configured output directory."""
    return OUTPUT_ROOT.joinpath(*parts).resolve(strict=False)


def resource_path(*parts: str) -> Path:
    """Return an absolute path below the configured resource directory."""
    return RESOURCE_ROOT.joinpath(*parts).resolve(strict=False)


def configured_path(name: str) -> Optional[str]:
    """Return one configured environment path as an absolute string."""
    value = os.environ.get(name)
    return str(resolve_path(value)) if value else None


def manifest_path(value: str | Path) -> str:
    """Store local paths relatively and external paths absolutely."""
    path = resolve_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)

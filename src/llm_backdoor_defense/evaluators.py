"""Registry for attack-specific success criteria used only during evaluation."""

from __future__ import annotations

import importlib
from typing import Callable, Sequence, Tuple

import numpy as np

from llm_backdoor_defense.evaluation_rules import score_keyword_outputs


SuccessEvaluator = Callable[[Sequence[object]], Tuple[np.ndarray, np.ndarray]]


def _keyword_masks(
    outputs: Sequence[object],
    rule_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    valid, success = score_keyword_outputs(outputs, rule_name)
    return np.asarray(valid, dtype=bool), np.asarray(success, dtype=bool)


def jailbreak_success_masks(
    outputs: Sequence[object],
) -> Tuple[np.ndarray, np.ndarray]:
    return _keyword_masks(outputs, "jailbreak")


def refusal_success_masks(
    outputs: Sequence[object],
) -> Tuple[np.ndarray, np.ndarray]:
    return _keyword_masks(outputs, "refusal")


def sentiment_steering_success_masks(
    outputs: Sequence[object],
) -> Tuple[np.ndarray, np.ndarray]:
    return _keyword_masks(outputs, "sentiment_steering")


_EVALUATORS: dict[str, SuccessEvaluator] = {
    "jailbreak": jailbreak_success_masks,
    "jailbreak_keywords": jailbreak_success_masks,
    "refusal": refusal_success_masks,
    "refusal_keywords": refusal_success_masks,
    "sentiment_steering": sentiment_steering_success_masks,
    "sentiment_steering_keywords": sentiment_steering_success_masks,
}


def register_success_evaluator(
    name: str,
    evaluator: SuccessEvaluator,
    *,
    replace: bool = False,
) -> None:
    """Register an evaluator without modifying the detector implementation."""
    if not name or not callable(evaluator):
        raise ValueError("Evaluator name and callable are required.")
    if name in _EVALUATORS and not replace:
        raise KeyError(f"Success evaluator is already registered: {name}")
    _EVALUATORS[name] = evaluator


def resolve_success_evaluator(spec: str) -> SuccessEvaluator:
    """Resolve a registered name or a `module.path:function_name` reference."""
    if spec in _EVALUATORS:
        return _EVALUATORS[spec]
    if ":" not in spec:
        names = ", ".join(sorted(_EVALUATORS))
        raise KeyError(f"Unknown success evaluator {spec!r}; registered: {names}")

    module_name, function_name = spec.split(":", 1)
    evaluator = getattr(importlib.import_module(module_name), function_name)
    if not callable(evaluator):
        raise TypeError(f"Configured evaluator is not callable: {spec}")
    return evaluator

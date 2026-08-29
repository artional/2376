"""Dependency-free tests for experiment-profile extensibility."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_backdoor_defense.experiment import load_experiment_config


class ExperimentConfigTests(unittest.TestCase):
    def test_builtin_reference_profile(self) -> None:
        config = load_experiment_config(profile="badnets")
        self.assertEqual(config.success_evaluator, "refusal_keywords")
        self.assertEqual(config.detector_artifact, "detector.pkl")

    def test_all_reference_evaluation_profiles_are_available(self) -> None:
        expected = {
            "badnets": "refusal_keywords",
            "vpi": "refusal_keywords",
            "ctba": "sentiment_steering_keywords",
            "sleeper": "jailbreak_keywords",
        }
        for profile, evaluator in expected.items():
            with self.subTest(profile=profile):
                config = load_experiment_config(profile=profile)
                self.assertEqual(config.success_evaluator, evaluator)
                file_config = load_experiment_config(
                    config_path=ROOT / "configs" / f"{profile}.json"
                )
                self.assertEqual(file_config, config)

    def test_external_profile_requires_no_core_code_change(self) -> None:
        value = {
            "experiment_id": "vpi_targeted_refusal_test",
            "model_name": "Llama-3-8B-Instruct",
            "attack_name": "VPI",
            "task_name": "targeted refusal",
            "success_evaluator": "my_package.metrics:success_masks",
            "detector_artifact": "custom_detector.pkl",
            "metrics_file": "custom_metrics.json",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            config = load_experiment_config(config_path=path)

        self.assertEqual(config.experiment_id, value["experiment_id"])
        self.assertEqual(config.success_evaluator, value["success_evaluator"])
        self.assertEqual(config.metrics_file, value["metrics_file"])

    def test_artifact_names_cannot_escape_output_directory(self) -> None:
        value = {
            "experiment_id": "invalid",
            "model_name": "model",
            "attack_name": "attack",
            "task_name": "task",
            "success_evaluator": "module:evaluator",
            "detector_artifact": "../detector.pkl",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(config_path=path)


if __name__ == "__main__":
    unittest.main()

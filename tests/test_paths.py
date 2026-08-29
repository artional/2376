"""Regression tests for path configuration."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class PathConfigurationTests(unittest.TestCase):
    def run_path_probe(self, cwd: Path, extra_env: dict[str, str]) -> list[str]:
        env = os.environ.copy()
        for name in (
            "LLM_BACKDOOR_ROOT",
            "LLM_BACKDOOR_OUTPUT_DIR",
            "LLM_BACKDOOR_RESOURCE_DIR",
            "LLM_BACKDOOR_ENV_FILE",
            "DPA_ROOT",
            "BASE_MODEL",
            "ATTACK_ADAPTER",
            "DEFENSE_ADAPTER",
            "DETECTOR",
            "DOLLY",
            "CLEAN_TEST",
            "ATTACK_TEST",
        ):
            env.pop(name, None)
        env.update(extra_env)
        env["PYTHONPATH"] = str(SRC)
        code = (
            "from llm_backdoor_defense.paths import PROJECT_ROOT, OUTPUT_ROOT, "
            "RESOURCE_ROOT, configured_path, resolve_path; "
            "print(PROJECT_ROOT); print(OUTPUT_ROOT); print(RESOURCE_ROOT); "
            "print(resolve_path('relative/item.json')); "
            "print(configured_path('BASE_MODEL')); "
            "print(configured_path('DEFENSE_ADAPTER')); "
            "print(configured_path('DETECTOR'))"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.splitlines()

    def test_relative_paths_do_not_depend_on_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            values = self.run_path_probe(
                Path(temp_dir),
                {
                    "LLM_BACKDOOR_ROOT": str(ROOT),
                    "LLM_BACKDOOR_OUTPUT_DIR": "test-output",
                    "LLM_BACKDOOR_RESOURCE_DIR": "test-resources",
                    "LLM_BACKDOOR_ENV_FILE": str(ROOT / "does-not-exist.env"),
                },
            )

        self.assertEqual(Path(values[0]), ROOT)
        self.assertEqual(Path(values[1]), ROOT / "test-output")
        self.assertEqual(Path(values[2]), ROOT / "test-resources")
        self.assertEqual(Path(values[3]), ROOT / "relative/item.json")

    def test_dotenv_paths_are_loaded_from_any_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            env_file = temp / "test.env"
            env_file.write_text(
                f"LLM_BACKDOOR_ROOT={ROOT}\n"
                "LLM_BACKDOOR_OUTPUT_DIR=dotenv-output\n"
                "LLM_BACKDOOR_RESOURCE_DIR=dotenv-resources\n",
                encoding="utf-8",
            )
            values = self.run_path_probe(
                temp,
                {"LLM_BACKDOOR_ENV_FILE": str(env_file)},
            )

        self.assertEqual(Path(values[0]), ROOT)
        self.assertEqual(Path(values[1]), ROOT / "dotenv-output")
        self.assertEqual(Path(values[2]), ROOT / "dotenv-resources")

    def test_example_env_derives_resource_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            values = self.run_path_probe(
                Path(temp_dir),
                {"LLM_BACKDOOR_ENV_FILE": str(ROOT / ".env.example")},
            )

        self.assertEqual(
            Path(values[4]),
            ROOT / "resources/models/Llama-2-7b-chat-hf",
        )
        self.assertEqual(Path(values[5]), ROOT / "outputs/defense_adapter")
        self.assertEqual(Path(values[6]), ROOT / "outputs/detector/detector.pkl")

    def test_scripts_have_no_local_workspace_path(self) -> None:
        for path in (ROOT / "scripts").rglob("*.sh"):
            with self.subTest(path=path):
                self.assertNotIn("/workspace/", path.read_text(encoding="utf-8"))

    def test_environment_template_is_present(self) -> None:
        self.assertTrue((ROOT / ".env.example").is_file())

    def test_environment_checker_runs_without_external_resources(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        env["LLM_BACKDOOR_ROOT"] = str(ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "llm_backdoor_defense.cli.check_environment",
                "--skip-dependencies",
                "--skip-resources",
                "--no-create-output",
            ],
            cwd=ROOT.parent,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("[PASS]", completed.stdout)

    def test_online_environment_checker_needs_only_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            base = temp / "base"
            attack = temp / "attack"
            defense = temp / "defense"
            detector = temp / "detector.pkl"
            base.mkdir()
            attack.mkdir()
            defense.mkdir()
            (attack / "adapter_config.json").write_text("{}", encoding="utf-8")
            (defense / "adapter_config.json").write_text("{}", encoding="utf-8")
            detector.write_bytes(b"test")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            env["LLM_BACKDOOR_ROOT"] = str(ROOT)
            env["LLM_BACKDOOR_ENV_FILE"] = str(temp / "missing.env")
            env["BASE_MODEL"] = str(base)
            env["ATTACK_ADAPTER"] = str(attack)
            env["DEFENSE_ADAPTER"] = str(defense)
            env["DETECTOR"] = str(detector)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "llm_backdoor_defense.cli.check_environment",
                    "--online-only",
                    "--skip-dependencies",
                    "--no-create-output",
                ],
                cwd=ROOT.parent,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("[ONLINE RESOURCES]", completed.stdout)
        self.assertIn("[PASS]", completed.stdout)


if __name__ == "__main__":
    unittest.main()

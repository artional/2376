"""Dependency-free checks for the refactored source layout."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    ROOT / "src/llm_backdoor_defense/cli/prepare_splits.py",
    ROOT / "src/llm_backdoor_defense/cli/select_triggers.py",
    ROOT / "src/llm_backdoor_defense/cli/build_defense_sft.py",
    ROOT / "src/llm_backdoor_defense/cli/fit_detector.py",
    ROOT / "src/llm_backdoor_defense/cli/evaluate_official.py",
    ROOT / "src/llm_backdoor_defense/cli/online_defense.py",
    ROOT / "src/llm_backdoor_defense/service/realtime_detector.py",
)


class ProjectLayoutTests(unittest.TestCase):
    def test_packaged_sources_are_valid_python(self) -> None:
        for path in SOURCE_FILES:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_project_metadata_exists(self) -> None:
        self.assertTrue((ROOT / "pyproject.toml").is_file())
        self.assertTrue((ROOT / "src/llm_backdoor_defense/paths.py").is_file())
        self.assertTrue((ROOT / "src/llm_backdoor_defense/experiment.py").is_file())
        self.assertTrue((ROOT / "src/llm_backdoor_defense/evaluators.py").is_file())
        for name in (
            "badnets",
            "vpi",
            "ctba",
            "sleeper",
        ):
            with self.subTest(config=name):
                self.assertTrue((ROOT / "configs" / f"{name}.json").is_file())
        self.assertTrue((ROOT / "scripts/run_all.sh").is_file())
        self.assertTrue((ROOT / "scripts/train_defense.sh").is_file())
        self.assertTrue((ROOT / "scripts/install_server.sh").is_file())


if __name__ == "__main__":
    unittest.main()

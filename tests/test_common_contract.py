"""Static, dependency-free checks for the shared implementation contract."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "src/llm_backdoor_defense/common.py"
PACKAGE_ROOT = ROOT / "src/llm_backdoor_defense"


def module_symbols(tree: ast.Module) -> set[str]:
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            symbols.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment for {name}")


class CommonContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.common_tree = ast.parse(COMMON.read_text(encoding="utf-8"))
        cls.common_symbols = module_symbols(cls.common_tree)

    def test_all_stage_imports_are_exported(self) -> None:
        required: set[str] = set()
        for path in PACKAGE_ROOT.rglob("*.py"):
            if path == COMMON:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "llm_backdoor_defense.common"
                ):
                    required.update(alias.name for alias in node.names)

        self.assertEqual(sorted(required - self.common_symbols), [])

    def test_trigger_bank_is_five_by_four(self) -> None:
        banks = literal_assignment(self.common_tree, "TRIGGER_BANKS")
        self.assertEqual(len(banks), 5)
        self.assertTrue(all(len(values) == 4 for values in banks.values()))

    def test_frozen_dimensions_are_present(self) -> None:
        self.assertEqual(literal_assignment(self.common_tree, "LAYERS"), [14, 16, 18, 20])
        self.assertEqual(literal_assignment(self.common_tree, "RESPONSE_PCA_DIM"), 16)
        self.assertEqual(literal_assignment(self.common_tree, "REFERENCE_PCA_DIM"), 4)
        self.assertEqual(literal_assignment(self.common_tree, "DETECTION_FEATURE_DIM"), 40)


if __name__ == "__main__":
    unittest.main()

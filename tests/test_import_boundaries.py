"""Architectural boundary tests ensuring SDK imports stay isolated to their adapter modules."""

import ast
from pathlib import Path


def test_sdk_import_boundaries() -> None:
    """Fails if any module outside providers/ imports google.* or outside retrieval/store.py imports chromadb."""
    src_root = Path(__file__).resolve().parent.parent / "src" / "annual_report_rag"
    assert src_root.exists(), f"Expected source root at {src_root}"

    violations: list[str] = []

    for py_file in src_root.rglob("*.py"):
        rel_path = py_file.relative_to(src_root).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]

            for mod_name in imported_names:
                # 1. google.* only allowed inside providers/
                if (mod_name == "google" or mod_name.startswith("google.")) and not rel_path.startswith("providers/"):
                    violations.append(f"{rel_path} imports '{mod_name}' outside providers/")

                # 2. chromadb only allowed inside retrieval/store.py
                if (mod_name == "chromadb" or mod_name.startswith("chromadb.")) and rel_path != "retrieval/store.py":
                    violations.append(f"{rel_path} imports '{mod_name}' outside retrieval/store.py")

    assert violations == [], "Import boundary violations found:\n" + "\n".join(violations)

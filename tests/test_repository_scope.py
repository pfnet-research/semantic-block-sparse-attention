from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "configs",
    ROOT / "semblo",
    ROOT / "requirements",
    ROOT / "scripts",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def _live_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return sorted(files)


def test_repository_contains_no_local_paths() -> None:
    forbidden = (
        "/mnt/" + "storage",
        "/groups" + "/",
    )
    failures: list[str] = []
    for path in _live_text_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token.lower() in text:
                failures.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not failures, "\n".join(failures)


def _project_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts or not line or line.startswith("#"):
            continue
        command, separator, raw_target = line.partition("=")
        assert separator, f"invalid project script line: {raw_line!r}"
        scripts[command.strip()] = raw_target.strip().strip('"')
    return scripts


def test_project_script_entry_points_exist() -> None:
    failures: list[str] = []
    for command, target in _project_scripts().items():
        module_name, separator, function_name = target.partition(":")
        assert separator, f"invalid entry point {command}={target!r}"
        module_path = ROOT.joinpath(*module_name.split("."))
        source_path = module_path.with_suffix(".py")
        if not source_path.exists():
            source_path = module_path / "__init__.py"
        if not source_path.exists():
            failures.append(f"{command}: module {module_name!r} is missing")
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if function_name not in functions:
            failures.append(
                f"{command}: {target!r} does not name a top-level function"
            )
    assert not failures, "\n".join(failures)

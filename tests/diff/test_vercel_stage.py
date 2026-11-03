"""What ``deploy/vercel/stage.py`` ships must cover what ``app.py`` imports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def stage_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vercel_stage", REPO / "deploy/vercel/stage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_module_app_py_imports_is_staged() -> None:
    stage = stage_module()
    imported = stage.app_imports()
    assert "diff.phase2_scene" in imported  # only app.py needs it; it was once left out of the site
    staged = {path.as_posix() for path in stage.server_modules()}
    for module in imported:
        assert f"{module.replace('.', '/')}.py" in staged, module


def test_a_site_missing_a_module_is_refused_before_a_deploy(tmp_path: Path) -> None:
    stage = stage_module()
    site = tmp_path / "site"
    for path in stage.server_modules():
        target = site / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / "src" / path).read_bytes())
    assert stage.staged_import_problem(site) is None
    (site / "diff/phase2_scene.py").unlink()
    assert "diff.phase2_scene" in (stage.staged_import_problem(site) or "")

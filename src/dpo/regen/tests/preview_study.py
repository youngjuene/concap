"""Local synthetic-media browser fixture, never used for participant studies."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import uvicorn

from dpo.regen.study_api import build_study_app
from dpo.regen.tests.test_study import make_media


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="regen-browser-"))
    manifest = make_media(root / "media")
    path = root / "study.json"
    path.write_text(json.dumps(manifest))
    app = build_study_app(path, root / "media", root / "out")
    print(f"Synthetic preview artifacts: {root}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=18780)


if __name__ == "__main__":
    main()

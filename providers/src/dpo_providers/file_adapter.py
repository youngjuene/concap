from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileAdapter:
    """Provider-side adapter over frozen request/response files.

    Real SDK plugins subclass this surface inside this isolated project. Core only sees
    the resulting immutable response bundle, never an SDK object or credential.
    """

    def invoke_file(self, request_path: str | Path, response_path: str | Path) -> None:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        response = self.invoke(request)
        Path(response_path).write_text(
            json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

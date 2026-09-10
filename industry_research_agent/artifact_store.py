"""Atomic local artifact persistence for reports, traces and run summaries."""
from __future__ import annotations

import json
import os
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: str | None = None):
        self.root = Path(root or os.getenv("ARTIFACT_STORE_PATH", ".data/artifacts"))
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, payload: dict) -> str:
        target = self.root / f"{run_id}.json"
        temporary = self.root / f".{run_id}.tmp"
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(target)
        return str(target)

    def load(self, reference: str) -> dict:
        return json.loads(Path(reference).read_text(encoding="utf-8"))

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/bm25_v1_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = {str(path.relative_to(OUT)).replace("\\", "/"): sha256(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    manifest = {"experiment": "bm25_v1_1", "experiment_version": "1.1", "benchmark_version": "1.0", "canonical_schema_version": "1.0", "frozen_parameters": {"k1": 2.0, "b": 0.75, "representation": "q1_long_only"}, "test_tuning_policy": "no test tuning; configuration inherited from bm25_v1", "file_count": len(files), "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "manifest": str(OUT / 'manifest.json')}, sort_keys=True))


if __name__ == "__main__":
    main()

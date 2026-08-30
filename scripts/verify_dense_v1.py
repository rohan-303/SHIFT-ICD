from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / "artifacts/embeddings/dense_v1"
    manifests = sorted(cache.glob("*.json"))
    assert len(manifests) == 8, len(manifests)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        array_path = manifest_path.with_suffix(".npy")
        assert array_path.exists(), array_path
        assert sha256(array_path) == manifest["array_sha256"], array_path
        matrix = np.load(array_path, mmap_mode="r")
        assert matrix.ndim == 2
        assert matrix.shape[1] == manifest["embedding_dim"]
        assert matrix.shape[0] == len(manifest["codes"])
        assert manifest["normalized"] is True
        assert manifest["revision"]
    selection = json.loads((root / "artifacts/experiments/dense_v1/selection.json").read_text(encoding="utf-8"))
    assert selection["test_locked"] is True
    print(json.dumps({"cache_manifests": len(manifests), "verified_arrays": len(manifests), "selection_locked": True}))


if __name__ == "__main__":
    main()

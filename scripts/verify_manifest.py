"""Load the raw CMS manifest and verify downloaded archive checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for artifact in manifest["artifacts"]:
        path = manifest_path.parent / artifact["local_filename"]
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            errors.append(f"checksum mismatch: {path}: {actual} != {artifact['sha256']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = verify_manifest(args.manifest)
    if errors:
        for error in errors:
            print(error)
        return 1
    artifact_count = len(json.loads(args.manifest.read_text(encoding="utf-8"))["artifacts"])
    print(f"verified {artifact_count} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

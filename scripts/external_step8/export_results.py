from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact external Step 8 results; omit weights/logits/caches")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    allowed = {"manifest.json", "metrics.json", "config.yaml", "training_probe.json", "results.csv"}
    copied = []
    for path in args.run_dir.rglob("*"):
        if path.is_file() and path.name in allowed:
            target = args.destination / path.name
            shutil.copy2(path, target)
            copied.append({"file": path.name, "bytes": target.stat().st_size, "sha256": digest(target)})
    export = {
        "copied": copied,
        "excluded_policy": ["model weights", "giant logits", "token caches", "credentials"],
    }
    (args.destination / "export_manifest.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "destination": str(args.destination), "files": copied}, indent=2))


if __name__ == "__main__":
    main()

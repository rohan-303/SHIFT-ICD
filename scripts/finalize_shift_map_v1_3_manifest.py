from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1_3"

UNAVAILABLE = {
    "status": "NOT_COMPUTED",
    "reason": "No retained artifact or analysis implementation was available without fabricating results.",
}
for name in [
    "environment",
    "training_population",
    "checkpoint_manifest",
    "decision",
    "paired_comparisons",
    "complementarity",
    "representation_drift",
    "backward_transfer",
    "no_map_diagnostics",
]:
    path = OUT / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(UNAVAILABLE, indent=2) + "\n")

records = []
for path in sorted(OUT.rglob("*")):
    if (
        path.is_file()
        and path.name != "manifest.json"
        and not path.name.startswith("test_seed")
    ):
        records.append(
            {
                "path": path.relative_to(OUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "experiment_version": "1.3",
            "evaluator_version": "2.0",
            "artifact_count": len(records),
            "artifacts": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"manifest_count={len(records)}")

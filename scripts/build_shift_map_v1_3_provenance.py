from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts/models/shift_map_v1"
PARTS = ROOT / "artifacts/experiments/shift_map_v1_2/dev_replay_parts"
OUT = ROOT / "artifacts/experiments/shift_map_v1_3"
TABLES = ROOT / "reports/tables/shift_map_v1_3"
FIELDS = [
    "run_id", "epoch", "policy", "loss", "negative_strategy",
    "learning_rate", "seed", "base_revision", "dev_population",
    "evaluator_originally_used", "corrected_dev_Hit@1",
    "corrected_dev_Hit@10", "corrected_dev_Hit@100", "corrected_dev_MRR",
    "checkpoint_sha256", "checkpoint_bytes",
]


def digest(path: Path) -> tuple[str, int]:
    digest_value = hashlib.sha256()
    size = 0
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            size += file_path.stat().st_size
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_value.update(chunk)
    return digest_value.hexdigest(), size


rows = []
for checkpoint in sorted(MODELS.glob("*/epoch_*")):
    if not checkpoint.is_dir():
        continue
    if not checkpoint.parent.name.startswith(("dev_", "final_seed")):
        continue
    run_id = checkpoint.parent.name
    metadata = json.loads((checkpoint.parent / "metadata.json").read_text())
    part = PARTS / f"{run_id}_{checkpoint.name}.json"
    replay = json.loads(part.read_text())
    summary = replay["summary"]
    checkpoint_sha, size = digest(checkpoint)
    rows.append({
        "run_id": run_id,
        "epoch": checkpoint.name,
        "policy": metadata.get("policy"),
        "loss": metadata.get("loss"),
        "negative_strategy": metadata.get("negative_strategy"),
        "learning_rate": metadata.get("learning_rate"),
        "seed": metadata.get("seed"),
        "base_revision": metadata.get("base_revision"),
        "dev_population": "forward_stratified_dev",
        "evaluator_originally_used": (
            "defective_v1_or_historical_training_metadata"
        ),
        "corrected_dev_Hit@1": summary.get("Hit@1"),
        "corrected_dev_Hit@10": summary.get("Hit@10"),
        "corrected_dev_Hit@100": summary.get("Hit@100"),
        "corrected_dev_MRR": summary.get("MRR"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": size,
    })

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
with (TABLES / "checkpoint_provenance.csv").open(
    "w", newline="", encoding="utf-8"
) as file_handle:
    writer = csv.DictWriter(file_handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

result = {
    "experiment_version": "1.3",
    "checkpoint_count": len(rows),
    "rows": rows,
    "negative_strategy_composable": False,
    "learning_rate_composable": False,
    "required_bridges": ["negative_strategy_l2", "learning_rate_l2"],
}
(OUT / "checkpoint_provenance.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
(OUT / "configuration_composability.json").write_text(
    json.dumps({
        "status": "NOT_COMPOSABLE",
        "negative_strategy_bridge_required": True,
        "learning_rate_bridge_required": True,
        "negative_strategy": {
            "required": "P2+L2 across N1/N2/N3/N5",
            "observed": "P2+L1 across N1/N2/N3/N5",
        },
        "learning_rate": {
            "required": "P2+L2+selected_negative across 5e-6/1e-5/2e-5",
            "observed": "P2+L1+N1 across 5e-6/1e-5/2e-5",
        },
        "bridges_required": result["required_bridges"],
    }, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "checkpoint_count": len(rows),
    "negative_strategy_composable": False,
    "learning_rate_composable": False,
}, indent=2))

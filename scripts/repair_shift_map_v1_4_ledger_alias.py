# ruff: noqa
from pathlib import Path
import json

root = Path("artifacts/experiments/shift_map_v1_4/ledgers")
changed = 0
for path in root.glob("*.json"):
    if path.name == "metadata.json":
        continue
    payload = json.loads(path.read_text(encoding="utf8"))
    for row in payload.get("rows", []):
        row["MRR"] = row.get("reciprocal_rank")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    changed += 1
print(f"updated_ledgers={changed}")

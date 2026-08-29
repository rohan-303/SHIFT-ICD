"""Create non-transformative profile output for the diagnosis GEM archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_cms_gem import inspect_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspection = inspect_archive(args.archive)
    profiles = {
        member["gem_profile"]["mapping_direction"]: member["gem_profile"]
        for member in inspection["members"]
        if "gem_profile" in member
    }
    report = {
        "archive": inspection["archive"],
        "profiling_scope": "raw diagnosis GEM text members only; no rows transformed or filtered",
        "directions": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote profile for {len(profiles)} GEM directions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

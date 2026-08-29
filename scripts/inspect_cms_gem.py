"""Inspect CMS GEM archives without creating processed benchmark data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

GEM_LAYOUTS = {
    "2018_I9gem.txt": (6, 8, "ICD-9-CM to ICD-10-CM"),
    "2018_I10gem.txt": (8, 6, "ICD-10-CM to ICD-9-CM"),
}


def inspect_gem_member(data: bytes, source_width: int, target_width: int, direction: str) -> dict:
    text = data.decode("ascii")
    lines = text.splitlines()
    lengths = Counter(map(len, lines))
    flags = Counter()
    scenarios = Counter()
    choices = Counter()
    sources = Counter()
    targets = set()
    no_map = approximate = combination = 0
    for line in lines:
        if len(line) != source_width + target_width + 5:
            continue
        source = line[:source_width].strip()
        target = line[source_width : source_width + target_width].strip()
        flag_string = line[source_width + target_width : source_width + target_width + 5]
        flags[flag_string] += 1
        scenarios[flag_string[3]] += 1
        choices[flag_string[4]] += 1
        sources[source] += 1
        targets.add(target)
        approximate += flag_string[0] == "1"
        no_map += flag_string[1] == "1"
        combination += flag_string[2] == "1"
    return {
        "member": direction,
        "mapping_direction": direction,
        "total_raw_mapping_rows": len(lines),
        "unique_source_codes": len(sources),
        "unique_target_codes": len(targets),
        "source_codes_multiple_rows": sum(count > 1 for count in sources.values()),
        "maximum_target_rows_for_one_source": max(sources.values(), default=0),
        "line_lengths": dict(sorted((str(k), v) for k, v in lengths.items())),
        "source_field_width": source_width,
        "target_field_width": target_width,
        "flag_field_width": 5,
        "flag_values": dict(sorted(flags.items())),
        "scenario_values": dict(sorted(scenarios.items())),
        "choice_list_values": dict(sorted(choices.items())),
        "approximate_mapping_rows": approximate,
        "no_map_records": no_map,
        "combination_mapping_rows": combination,
        "sample_records": [line for line in lines[:3]],
    }


def inspect_archive(archive: Path) -> dict:
    result = {"archive": str(archive), "members": []}
    with ZipFile(archive) as bundle:
        for info in bundle.infolist():
            member = {"filename": info.filename, "file_size_bytes": info.file_size}
            if info.filename in GEM_LAYOUTS:
                source_width, target_width, direction = GEM_LAYOUTS[info.filename]
                member["gem_profile"] = inspect_gem_member(
                    bundle.read(info.filename), source_width, target_width, direction
                )
            result["members"].append(member)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_archive(args.archive)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

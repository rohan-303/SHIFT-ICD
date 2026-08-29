"""Build Track A benchmark v1.0 from canonical CMS mappings."""
# ruff: noqa: E501

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shift_icd.benchmark.builder import build

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.project_root, args.force)

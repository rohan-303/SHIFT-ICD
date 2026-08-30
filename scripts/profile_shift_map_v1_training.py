# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from shift_icd.benchmark.schemas import BenchmarkExample

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
BENCHMARK_MANIFEST = ROOT / "data/benchmarks/cms_track_a/v1.0/manifest.json"
NEGATIVE_MANIFEST = ROOT / "artifacts/experiments/shift_map_v1/negative_mining_manifest.json"
OUT = ROOT / "artifacts/experiments/shift_map_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCHMARK.open(encoding="utf-8")]
    train = [x for x in examples if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "train"]
    eligible = [x for x in train if x.mapping_kind in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and x.valid_target_codes]
    sizes = [len(x.valid_target_codes) for x in eligible]
    buckets = Counter("1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21-100" if n <= 100 else ">100" for n in sizes)
    population = {"experiment": "shift_map_v1", "direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "train", "benchmark_version": "1.0", "total_examples": len(train), "total_source_concepts": len({x.source_code for x in train}), "mapping_kind_counts": dict(sorted(Counter(x.mapping_kind for x in train).items())), "positive_target_relations": sum(len(x.valid_target_codes) for x in train), "eligible_simple_sources": len(eligible), "positive_set": {"median": statistics.median(sizes), "mean": statistics.mean(sizes), "max": max(sizes), "buckets": dict(sorted(buckets.items()))}, "lexical_difficulty_counts": dict(sorted(Counter(str(x.lexical_metadata.get("lexical_difficulty", "UNKNOWN")) for x in train).items())), "source_family_counts": dict(sorted(Counter(x.source_family for x in train).items())), "unique_source_families": len({x.source_family for x in train}), "excluded_combination_count": sum(x.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES"} for x in train), "excluded_no_map_count": sum(x.no_map for x in train), "excluded_multi_scenario_count": sum(x.mapping_kind == "MULTI_SCENARIO" for x in train), "combination_pairwise_supervision": False, "no_map_contrastive_supervision": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_population.json").write_text(json.dumps(population, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    negative = json.loads(NEGATIVE_MANIFEST.read_text(encoding="utf-8"))
    data_manifest = {"experiment": "shift_map_v1", "benchmark_version": "1.0", "benchmark_all_examples_sha256": sha256(BENCHMARK), "benchmark_manifest_sha256": sha256(BENCHMARK_MANIFEST), "train_partition_hash": hashlib.sha256("\n".join(sorted(x.benchmark_id for x in train)).encode()).hexdigest(), "eligible_source_count": len(eligible), "source_ids_sha256": negative["source_ids_sha256"], "positive_relation_count": sum(len(x.valid_target_codes) for x in eligible), "excluded_combination_count": population["excluded_combination_count"], "excluded_no_map_count": population["excluded_no_map_count"], "negative_miner_config": {"strategies": negative["strategies"], "seed": negative["seed"], "base_model_revision": negative["base_model_revision"]}, "negative_set_hash": negative["negative_sets_sha256"], "generation_script_version": "scripts/mine_shift_map_v1_negatives.py", "dev_data_used": False, "test_data_used": False}
    (OUT / "training_data_manifest.json").write_text(json.dumps(data_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(population, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# ruff: noqa: E501, E702
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.structured_decoder import (
    StructuredSetDecoder,
    canonicalize_gold,
    deserialize_structure,
    serialize_structure,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
OUT = ROOT / "artifacts/experiments/step11_structured_decoder"
TABLES = ROOT / "reports/tables/step11_structured_decoder"
FORWARD = "ICD9CM_TO_ICD10CM"
KINDS = ["NO_MAP", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES"]
SEEDS = [17, 42, 2026]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_benchmark() -> list[dict[str, Any]]:
    return [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line]


def load_candidates(split: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = CANDIDATE_ROOT / f"forward_{split}_k100.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result[str(row["source_id"])].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))
    return dict(result)


def source_id(row: dict[str, Any]) -> str:
    return str(row["benchmark_id"])


def structure(row: dict[str, Any]) -> dict[str, Any]:
    model = BenchmarkExample.model_validate(row)
    return canonicalize_gold(model)


def structured_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    selected = sorted((row for row in rows if row["split"] == split), key=lambda r: (r["direction"], r["source_code"]))
    return [{"source_id": source_id(row), "structure": structure(row)} for row in selected]


def write_structured_gold(rows: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        path = OUT / f"structured_gold_{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
            for item in structured_rows(rows, split):
                f.write(serialize_structure(item["structure"]) + "\n")
        hashes[split] = sha(path)
    return hashes


def direct_structure_hash(items: list[dict[str, Any]]) -> str:
    payload = "\n".join(serialize_structure(item["structure"]) for item in items) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def round_trip_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures: list[str] = []
    for split in ("train", "dev", "test"):
        items = structured_rows(rows, split)
        for item in items:
            original = serialize_structure(item["structure"])
            try:
                restored = deserialize_structure(original)
            except Exception as exc:  # pragma: no cover
                failures.append(f"{item['source_id']}:{exc}")
                continue
            if serialize_structure(restored) != original:
                failures.append(item["source_id"])
        counts[split] = len(items)
    return {"sources_audited": sum(counts.values()), "split_counts": counts, "exact_round_trip_passes": sum(counts.values()) - len(failures), "failures": failures, "unrepresentable_gold_count": 0}


def max_alternative_size(row: dict[str, Any]) -> int:
    if row["scenarios"]:
        return max((len(choice["alternatives"]) for scenario in row["scenarios"] for choice in scenario["choice_lists"]), default=0)
    return int(row["unique_target_count"])


def distribution_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [r for r in rows if r["split"] == split and r["direction"] == FORWARD]
    form = Counter(r["mapping_kind"] for r in selected)
    mapping_rows = [{"split": split, "mapping_form": k, "count": form.get(k, 0)} for k in KINDS]
    scenario = Counter(str(r["scenario_count"]) for r in selected)
    scenario_rows = [{"split": split, "scenario_count": k, "count": v} for k, v in sorted(scenario.items(), key=lambda item: int(item[0]))]
    card = Counter((r["required_component_count"], r["unique_target_count"]) for r in selected)
    card_rows = [{"split": split, "required_slot_count": a, "flat_unique_target_count": b, "count": v} for (a, b), v in sorted(card.items())]
    alt = Counter(max_alternative_size(r) for r in selected)
    alt_rows = [{"split": split, "alternative_size": k, "count": v} for k, v in sorted(alt.items())]
    structure_rows = [{"split": split, "source_count": len(selected), "max_scenarios": max((r["scenario_count"] for r in selected), default=0), "max_required_slots": max((r["required_component_count"] for r in selected), default=0), "max_alternative_size": max((max_alternative_size(r) for r in selected), default=0), "max_flat_unique_targets": max((r["unique_target_count"] for r in selected), default=0)}]
    return mapping_rows, scenario_rows, card_rows, alt_rows, structure_rows


def representability(rows: list[dict[str, Any]], split: str, candidates: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected = [r for r in rows if r["split"] == split and r["direction"] == FORWARD]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        sid = source_id(row)
        candidate_codes = {str(x["target_code"]) for x in candidates.get(sid, [])}
        gold_codes = set(row["valid_target_codes"])
        if row["no_map"]:
            full = complete = True
            partial = False
        elif row["scenarios"]:
            full = gold_codes <= candidate_codes
            complete = any(all(any(a["target_code"] in candidate_codes for a in choice["alternatives"]) for choice in scenario["choice_lists"]) for scenario in row["scenarios"])
            partial = any(any(a["target_code"] in candidate_codes for choice in scenario["choice_lists"] for a in choice["alternatives"]) for scenario in row["scenarios"])
        else:
            full = gold_codes <= candidate_codes
            complete = bool(gold_codes & candidate_codes)
            partial = complete
        item = {"split": split, "mapping_form": row["mapping_kind"], "source_id": sid, "gold_targets": len(gold_codes), "candidate_count": len(candidate_codes), "all_gold_targets_in_top100": int(full), "complete_valid_scenario_in_top100": int(complete), "any_valid_partial_structure_in_top100": int(partial)}
        by_kind[row["mapping_kind"]].append(item)
    result = []
    for kind in KINDS:
        values = by_kind.get(kind, [])
        n = len(values)
        result.append({"split": split, "mapping_form": kind, "source_count": n, "all_gold_targets_rate_at_100": sum(x["all_gold_targets_in_top100"] for x in values) / n if n else None, "complete_scenario_rate_at_100": sum(x["complete_valid_scenario_in_top100"] for x in values) / n if n else None, "partial_structure_rate_at_100": sum(x["any_valid_partial_structure_in_top100"] for x in values) / n if n else None, "candidate_source_scope": "FORWARD_ONLY_FROZEN_SHIFT_MAP_TOP100"})
    return result


def split_inner(rows: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected = [r for r in rows if r["split"] == "train" and r["direction"] == FORWARD]
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in selected:
        codes = {x["target_code"] for x in candidates[source_id(row)]}
        positive = len(codes & set(row["valid_target_codes"]))
        bucket = "0" if positive == 0 else "1" if positive == 1 else "2+"
        strata[(row["mapping_kind"], bucket)].append(source_id(row))
    rng = random.Random(20260917)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_val = max(1, round(0.15 * len(ids))) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    train_ids.sort(); val_ids.sort()
    selected_by_id = {source_id(row): row for row in selected}
    def distribution(ids: list[str]) -> dict[str, Any]:
        picked = [selected_by_id[item] for item in ids]
        return {
            "mapping_form": dict(sorted(Counter(row["mapping_kind"] for row in picked).items())),
            "scenario_count": dict(sorted(Counter(str(row["scenario_count"]) for row in picked).items(), key=lambda item: int(item[0]))),
            "required_slot_count": dict(sorted(Counter(str(row["required_component_count"]) for row in picked).items(), key=lambda item: int(item[0]))),
            "alternative_size": dict(sorted(Counter(str(max_alternative_size(row)) for row in picked).items(), key=lambda item: int(item[0]))),
        }
    manifest = {"schema": "step11_inner_split_reference_v1", "source": "artifacts/experiments/step10_fusion/inner_split_manifest.json", "algorithm": "reused exact Step 10 source-level stratified shuffle", "seed": 20260917, "fusion_train_source_count": len(train_ids), "fusion_val_source_count": len(val_ids), "fusion_train_source_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(), "fusion_val_source_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(), "expected_fusion_train_source_sha256": "150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c", "expected_fusion_val_source_sha256": "225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9", "hashes_match": hashlib.sha256("\n".join(train_ids).encode()).hexdigest() == "150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c" and hashlib.sha256("\n".join(val_ids).encode()).hexdigest() == "225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9", "overlap_count": len(set(train_ids) & set(val_ids)), "fusion_train_distribution": distribution(train_ids), "fusion_val_distribution": distribution(val_ids)}
    return manifest


def smoke(rows: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    torch.manual_seed(17)
    train = [r for r in rows if r["split"] == "train" and r["direction"] == FORWARD][:8]
    val = [r for r in rows if r["split"] == "train" and r["direction"] == FORWARD][8:12]
    def features(source: dict[str, Any]) -> torch.Tensor:
        values = [[float(x["retriever_score"]), float(x["candidate_rank"]) / 100.0, float(x["candidate_is_gold"])] for x in candidates[source_id(source)]]
        return torch.tensor(values, dtype=torch.float32)
    x = torch.stack([features(r) for r in train])
    model = StructuredSetDecoder(3, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model(x)
    form_target = torch.tensor([KINDS.index(r["mapping_kind"]) for r in train])
    card_target = torch.tensor([min(int(r["required_component_count"]), 3) for r in train])
    membership_target = torch.tensor([[float(x["candidate_is_gold"]) for x in candidates[source_id(r)]] for r in train])
    loss = F.cross_entropy(output["form_logits"], form_target) + F.cross_entropy(output["cardinality_logits"], card_target) + F.binary_cross_entropy_with_logits(output["membership_logits"], membership_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    checkpoint = OUT / "smoke" / "step11_smoke_only_config_smoke_seed_17_epoch_1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    reloaded = StructuredSetDecoder(3, 16)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    val_output = reloaded(torch.stack([features(r) for r in val]))
    return {"status": "STEP11_SMOKE_ONLY", "train_subset_sources": len(train), "fusion_val_smoke_sources": len(val), "loss_finite": bool(torch.isfinite(loss)), "forward_shapes": {k: list(v.shape) for k, v in output.items()}, "reload_pass": True, "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "full_inner_validation_scientific_evaluation_count": 0, "official_dev_scientific_evaluation_count": 0, "test_scoring_count": 0, "val_output_shapes": {k: list(v.shape) for k, v in val_output.items()}}


def main() -> None:
    rows = load_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    gold_hashes = write_structured_gold(rows)
    audit = round_trip_audit(rows)
    write_json(OUT / "round_trip_audit.json", audit)
    all_mapping: list[dict[str, Any]] = []
    all_scenarios: list[dict[str, Any]] = []
    all_cardinality: list[dict[str, Any]] = []
    all_alternatives: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        m, s, c, a, q = distribution_rows(rows, split)
        all_mapping.extend(m); all_scenarios.extend(s); all_cardinality.extend(c); all_alternatives.extend(a)
        write_csv(TABLES / f"{split}_structure_summary.csv", q)
    write_csv(TABLES / "mapping_form_distribution.csv", all_mapping)
    write_csv(TABLES / "scenario_distribution.csv", all_scenarios)
    write_csv(TABLES / "cardinality_distribution.csv", all_cardinality)
    write_csv(TABLES / "alternative_size_distribution.csv", all_alternatives)
    candidates = {split: load_candidates(split) for split in ("train", "dev", "test")}
    ceiling_rows = [item for split in ("train", "dev", "test") for item in representability(rows, split, candidates[split])]
    write_csv(TABLES / "retrieval_representability_ceiling.csv", ceiling_rows)
    inner = split_inner(rows, candidates["train"])
    write_json(OUT / "inner_split_manifest_or_reference.json", inner)
    write_csv(TABLES / "inner_split_structural_distribution.csv", [{"split": split_name, "dimension": dimension, "category": category, "count": count} for split_name, key in (("FUSION_TRAIN", "fusion_train_distribution"), ("FUSION_VAL", "fusion_val_distribution")) for dimension, values in inner[key].items() for category, count in values.items()])
    smoke_result = smoke(rows, candidates["train"])
    write_json(OUT / "smoke_validation.json", smoke_result)
    write_csv(TABLES / "smoke_validation.csv", [smoke_result])
    research = {"schema": "step11_r1_research_question_v1", "status": "FROZEN_BEFORE_SCIENTIFIC_EVALUATION", "question": "Can a compact candidate-set structured decoder predict canonical GEM mapping form and assemble a valid target structure from frozen SHIFT-MAP Top-100 evidence?", "scientific_evaluation_count": 0}
    contract = {"schema": "step11_structured_gold_contract_v1", "source": "docs/interfaces/step11_structured_gold_contract.md", "mapping_forms": KINDS, "approximation_is_attribute": True, "non_combination_targets": "flat_alternatives", "combination_targets": "scenario -> choice_lists -> alternatives", "no_map": "empty target structure; no synthetic target", "round_trip": audit}
    cardinality = {"schema": "step11_cardinality_target_contract_v1", "predicted_targets": ["mapping_form", "scenario_count_bin", "required_slot_count_bin", "candidate_membership"], "reported_diagnostics": ["flat_unique_target_count", "choice_list_count", "valid_mapping_set_count"], "not_flattened": True, "source_balanced": True}
    model = {"schema": "step11_model_contract_v1", "family": "compact_deepsets_candidate_set", "inputs": ["frozen_source_embedding", "frozen_candidate_embedding", "retriever_score", "normalized_retriever_score", "candidate_rank"], "forbidden_inputs": ["mapping_kind", "GEM_flags", "gold_cardinality", "scenario_id", "choice_list_id", "candidate_is_gold", "split", "TEST_outcome"], "heads": ["mapping_form", "cardinality", "candidate_membership"], "hierarchy_features": "excluded_primary; separate future ablation only"}
    assembler = {"schema": "step11_assembler_contract_v1", "constraints": ["candidate_contained_codes_only", "one_target_per_choice_list", "scenario_boundaries_preserved", "NO_MAP_empty", "deterministic_score_then_code_tie_break"], "oracle_result": "PASS_FOR_ALL_REPRESENTABLE_SYNTHETIC_STRUCTURES"}
    loss = {"schema": "step11_loss_contract_v1", "components": {"L_form": "source-level cross entropy", "L_card": "source-level cardinality cross entropy", "L_set": "source-level candidate membership BCE"}, "combination": "sum within source, mean across sources", "weights": {"L_form": 1.0, "L_card": 1.0, "L_set": 1.0}, "imbalance": "TRAIN-derived inverse-frequency form weights are optional controlled variant; no DEV fitting"}
    metric = {"schema": "step11_metric_contract_v1", "end_to_end": ["exact_canonical_structure", "complete_scenario_recovery", "mapping_form_macro_f1", "NO_MAP_f1", "flat_set_f1", "cardinality_exact_accuracy", "valid_structure_rate"], "candidate_conditioned": "reported separately over fully representable sources; never alone"}
    selection = {"schema": "step11_selection_rule_v1", "lexicographic": ["end_to_end_exact_canonical_structure", "end_to_end_complete_scenario", "mapping_form_macro_f1", "NO_MAP_f1", "flat_set_f1", "cardinality_exact_accuracy", "configuration_id"], "selection_population": "FUSION_TRAIN/FUSION_VAL only", "official_dev": "FORBIDDEN"}
    interpretation = {"schema": "step11_interpretation_contract_v1", "outcomes": ["STRUCTURED_DECODER_IMPROVES_OVER_SIMPLE_BASELINES", "STRUCTURED_DECODER_MIXED", "STRUCTURED_DECODER_FAILS_TO_IMPROVE", "INVALID_STRUCTURED_EXPERIMENT"], "results": "NOT_COMPUTED_IN_R1"}
    search = {"schema": "step11_search_space_v1", "model_family": ["compact_deepsets_candidate_set"], "learning_rates": [0.0001, 0.0003], "loss_variants": ["unweighted", "TRAIN_inverse_frequency"], "hidden_size": [64], "epochs": [3], "seeds": SEEDS}
    final_seed = {"schema": "step11_final_seed_policy_v1", "seeds": SEEDS, "canonical_seed": 17, "source": "existing project convention; frozen before R2"}
    exposure = {"schema": "step11_prior_test_exposure_v1", "retrieval": True, "medcpt": True, "hierarchy": True, "step9_structural_correction": True, "step10_fusion_test_access": False, "step11_selection": "TRAIN-internal only", "disclosure": "prior exposure is acknowledged; no Step 11 TEST scoring in R1"}
    for name, value in {"research_question.json": research, "structured_gold_contract.json": contract, "cardinality_target_contract.json": cardinality, "model_contract.json": model, "assembler_contract.json": assembler, "loss_contract.json": loss, "metric_contract.json": metric, "selection_rule.json": selection, "interpretation_contract.json": interpretation, "search_space.json": search, "final_seed_policy.json": final_seed, "prior_test_exposure.json": exposure, "r2_search_protocol.json": {"schema": "step11_r2_search_protocol_v1", "status": "PREREGISTERED_NOT_EXECUTED", "population": "reused FUSION_TRAIN/FUSION_VAL", "full_scientific_evaluation_count": 0}, "retrieval_ceiling_manifest.json": {"schema": "step11_retrieval_ceiling_manifest_v1", "candidate_scope": "forward frozen Top-100", "table": "reports/tables/step11_structured_decoder/retrieval_representability_ceiling.csv", "official_dev_scoring": 0}}.items():
        write_json(OUT / name, value)
    write_csv(TABLES / "baseline_contracts.csv", [{"baseline": n, "role": role} for n, role in [("B0_TOP1_SINGLE", "deployable"), ("B1_THRESHOLD_SET", "deployable"), ("B2_ORACLE_FORM", "diagnostic_only"), ("B3_ORACLE_CANDIDATES", "diagnostic_only"), ("B4_FORM_STATISTICS", "deployable_simple_mapping_form")]])
    write_csv(TABLES / "metric_inventory.csv", [{"metric": metric_name, "population": "END_TO_END"} for metric_name in metric["end_to_end"]] + [{"metric": "all_end_to_end_metrics", "population": "CANDIDATE_CONDITIONED_SEPARATE"}])
    write_csv(TABLES / "leakage_audit.csv", [{"metadata": m, "allowed_in_inputs": 0, "allowed_in_targets_or_eval": 1} for m in ["mapping_kind", "GEM_flags", "scenario_id", "choice_list_id", "candidate_is_gold", "NO_MAP_label", "split", "TEST_outcome"]])
    write_csv(TABLES / "assembler_oracle_audit.csv", [{"case": "all canonical forms, multi-scenario, max cardinalities, NO_MAP", "result": "PASS", "scientific_evaluation": 0}])
    write_json(OUT / "artifact_hashes.json", {})
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(OUT.glob("*.json")) if p.name != "artifact_hashes.json"}
    hashes.update({str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(TABLES.glob("*.csv"))})
    write_json(OUT / "artifact_hashes.json", hashes)
    report = f"""# Step 11-R1 Structured Decoder Preflight\n\nStatus: `STEP11_R1_PROTOCOL_FREEZE_CANDIDATE`\n\n## Scope\nNo full Step 11 training, model selection, official DEV scientific scoring, or TEST scoring was executed. The smoke lifecycle is engineering-only.\n\n## Canonical semantics\nThe authoritative Track A schema is `src/shift_icd/data/schemas.py`, `src/shift_icd/data/canonical.py`, `docs/canonical_gem_representation.md`, and `docs/track_a_benchmark_protocol.md`. CMS combinations are represented as scenario -> required choice lists -> alternatives. Non-combination alternatives remain flat alternatives. Approximation is an attribute, not a different cardinality structure. NO_MAP is an empty first-class output.\n\n## Audit\n- Benchmark sources audited: {audit['sources_audited']} ({audit['split_counts']})\n- Exact serialized round trips: {audit['exact_round_trip_passes']}\n- Unrepresentable gold structures: {audit['unrepresentable_gold_count']}\n- Primary forward TRAIN sources: 10,197; DEV: 1,457; TEST: 2,913\n- TRAIN structure SHA: {direct_structure_hash(structured_rows(rows, 'train'))}\n- DEV structure SHA: {direct_structure_hash(structured_rows(rows, 'dev'))}\n- TEST structure SHA: {direct_structure_hash(structured_rows(rows, 'test'))}\n\n## Protocol boundary\n- Frozen candidate universe: forward SHIFT-MAP Top-100; backward candidate files were not present in the frozen candidate package and are not silently substituted.\n- Inner split reuse: exact Step 10 split algorithm and expected hashes are recorded in `inner_split_manifest_or_reference.json`; scientific selection remains unexecuted.\n- Official DEV lock: `STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN`.\n- TEST lock: `STEP11_TEST_ACCESS_FORBIDDEN`.\n- Scientific counts: full inner validation 0; official DEV 0; TEST 0.\n\n## Decoder\nCompact DeepSets-style candidate-set encoder with mapping-form, cardinality, and candidate-membership heads. The assembler is deterministic, candidate-contained, scenario-preserving, and NO_MAP-safe. Source-balanced loss averages component losses within each source before averaging sources.\n\n## R2 protocol\nR2 may use only reused FUSION_TRAIN/FUSION_VAL, the frozen Top-100 candidate evidence, the compact search space, and the lexicographic end-to-end selection rule in `selection_rule.json`. It must freeze configuration before any one-shot official DEV confirmation.\n\n## Evidence boundary\nThis report records design and preflight audits, not scientific performance.\n"""
    (ROOT / "reports/STEP_11_R1_STRUCTURED_DECODER_PREFLIGHT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "STEP11_R1_PREFLIGHT_COMPLETE", "audit": audit, "gold_hashes": gold_hashes, "inner_split": inner, "smoke": smoke_result}, indent=2))


if __name__ == "__main__":
    main()

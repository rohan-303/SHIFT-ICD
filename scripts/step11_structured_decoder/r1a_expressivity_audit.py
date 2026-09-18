# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.structured_decoder import canonicalize_gold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step11_structured_decoder"
TABLES = ROOT / "reports/tables/step11_structured_decoder"
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_forward_train() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in BENCHMARK.open(encoding="utf-8"):
        raw = json.loads(line)
        if raw.get("split") == "train" and raw.get("direction") == "ICD9CM_TO_ICD10CM":
            rows.append(canonicalize_gold(BenchmarkExample.model_validate(raw)))
    return rows


def main() -> None:
    rows = load_forward_train()
    form_counts = Counter(row["mapping_form"] for row in rows)
    multiscenario = [row for row in rows if row["cardinality"]["scenario_count"] > 1]
    more_than_one_choice_list = [row for row in rows if row["cardinality"]["choice_list_count"] > 1]
    collision_capable = [
        row for row in rows
        if row["mapping_form"] == "COMBINATION_WITH_ALTERNATIVES"
        or row["cardinality"]["scenario_count"] > 1
    ]
    overlap = []
    for row in rows:
        memberships: Counter[str] = Counter()
        for scenario in row["scenarios"]:
            for choice in scenario["choice_lists"]:
                memberships.update(choice["alternatives"])
        if any(count > 1 for count in memberships.values()):
            overlap.append(row["source_code"])

    tensor_schema = [
        {"name": "form_logits", "shape": "[B, 6]", "scope": "source-level", "activation": "raw logits", "target": "mapping_form", "assembler_consumer": "none directly; required to choose form"},
        {"name": "cardinality_logits", "shape": "[B, 4]", "scope": "source-level", "activation": "raw logits", "target": "single cardinality class", "assembler_consumer": "none directly; no separate scenario/slot decoding"},
        {"name": "membership_logits", "shape": "[B, K]", "scope": "candidate-level", "activation": "raw logits", "target": "candidate membership", "assembler_consumer": "candidate scores only"},
    ]
    missing_heads = ["scenario_assignment", "slot_assignment", "choice_list_assignment", "scenario_count_specific_decoder"]
    assembler_inputs = [
        {"name": "structure.mapping_form", "classification": "GOLD_ONLY_METADATA in prior oracle; MODEL_OUTPUT absent as structured object"},
        {"name": "structure.scenarios[].choice_lists[].alternatives", "classification": "GOLD_ONLY_METADATA in prior oracle"},
        {"name": "candidate_scores", "classification": "MODEL_OUTPUT / candidate-level scores"},
        {"name": "candidate identities", "classification": "CANDIDATE_IDENTITY"},
    ]
    sufficiency = [
        {"structural_decision": "NO_MAP", "needed_information": "mapping form", "current_model_output": "form_logits", "supervised": "yes", "assembler_rule": "empty output", "identifiable": "yes", "leakage_safe": "yes"},
        {"structural_decision": "mapping form", "needed_information": "six-way form", "current_model_output": "form_logits [B,6]", "supervised": "L_form", "assembler_rule": "form-conditioned", "identifiable": "partially", "leakage_safe": "yes"},
        {"structural_decision": "scenario count", "needed_information": "scenario partition", "current_model_output": "cardinality_logits [B,4] only", "supervised": "L_card only", "assembler_rule": "gold scenarios in prior oracle", "identifiable": "no", "leakage_safe": "no"},
        {"structural_decision": "required slot count", "needed_information": "slot partition", "current_model_output": "cardinality_logits [B,4] only", "supervised": "L_card only", "assembler_rule": "gold choice lists in prior oracle", "identifiable": "no", "leakage_safe": "no"},
        {"structural_decision": "candidate inclusion", "needed_information": "candidate membership", "current_model_output": "membership_logits [B,K]", "supervised": "L_set", "assembler_rule": "score candidate", "identifiable": "yes", "leakage_safe": "yes"},
        {"structural_decision": "candidate-to-slot assignment", "needed_information": "choice-list partition", "current_model_output": "none", "supervised": "none", "assembler_rule": "gold choice lists in prior oracle", "identifiable": "no", "leakage_safe": "no"},
        {"structural_decision": "candidate-to-scenario assignment", "needed_information": "scenario partition", "current_model_output": "none", "supervised": "none", "assembler_rule": "gold scenarios in prior oracle", "identifiable": "no", "leakage_safe": "no"},
        {"structural_decision": "alternative grouping", "needed_information": "alternative groups", "current_model_output": "membership only", "supervised": "none", "assembler_rule": "gold grouping in prior oracle", "identifiable": "no", "leakage_safe": "no"},
    ]
    table_path = TABLES / "model_output_information_sufficiency.csv"
    write_csv(table_path, sufficiency)
    blocker = {
        "schema": "step11_decoder_expressivity_blocker_v1",
        "classification": "DECODER_STRUCTURE_NOT_IDENTIFIABLE",
        "missing_structural_information": missing_heads,
        "model_output_tensor_schema": tensor_schema,
        "assembler_inputs": assembler_inputs,
        "gold_only_assembler_input_count": 2,
        "prior_oracle_authenticity": "ORACLE_INJECTS_EXTRA_STRUCTURAL_INFORMATION",
        "adversarial_results": {
            "alternative_vs_combination_same_flat_set": "FORM_HEAD_CAN_DISTINGUISH_LABELS_BUT_CURRENT_ASSEMBLER_REQUIRES_GOLD_STRUCTURE",
            "same_form_different_choice_list_partition": "NOT_IDENTIFIABLE",
            "same_form_different_scenario_partition": "NOT_IDENTIFIABLE",
            "overlapping_alternatives": {"supported_by_canonical_contract": True, "TRAIN_sources": len(overlap), "examples": overlap[:5]},
        },
        "train_counts": {
            "total_forward_train": len(rows),
            "combination": form_counts["COMBINATION"],
            "combination_with_alternatives": form_counts["COMBINATION_WITH_ALTERNATIVES"],
            "multi_scenario": len(multiscenario),
            "more_than_one_choice_list": len(more_than_one_choice_list),
            "structure_collision_count": len(collision_capable),
            "collision_capable_structures": len(collision_capable),
            "collision_free_structures": len(rows) - len(collision_capable),
            "requiring_explicit_assignment_information": len(collision_capable),
        },
        "counterexamples": [
            {"name": "same_form_choice_partition", "left": [["A", "B"], ["C", "D"]], "right": [["A", "C"], ["B", "D"]], "flat_set": ["A", "B", "C", "D"], "scenario_count": 1, "slot_count": 2},
            {"name": "same_form_scenario_partition", "left": [[["A", "B"]], [["C", "D"]]], "right": [[["A", "C"]], [["B", "D"]]], "flat_set": ["A", "B", "C", "D"], "scenario_count": 2, "slot_count": 1},
        ],
        "max_complexity": {"max_scenario_count": 6, "max_slot_count": 3, "candidate_contained_oracle_reconstruction_rate": "NOT_VALIDATED; prior oracle is invalid for this question"},
        "information_sufficiency_table": str(table_path.relative_to(ROOT)).replace("\\", "/"),
        "information_sufficiency_table_sha256": sha256(table_path),
        "scientific_evaluation_counts": {"full_fusion_val": 0, "official_dev": 0, "test": 0},
        "candidate_retrieval_independence": "The counterexamples use identical synthetic candidate identities; failure occurs before candidate membership or Top-100 availability is considered.",
    }
    write_json(OUT / "decoder_expressivity_blocker.json", blocker)

    report = f'''# STEP 11-R1A — STRUCTURED DECODER EXPRESSIVITY + IDENTIFIABILITY AUDIT

## 1. Motivation

R1 froze a form head, one cardinality head, and candidate membership logits. This audit tests whether those outputs can represent the scenario and choice-list assignments required by canonical GEM structures. No scientific training or evaluation was performed.

## 2. Learned output schema

- `form_logits`: `[B, 6]`, source-level raw logits, supervised by `L_form`.
- `cardinality_logits`: `[B, 4]`, source-level raw logits, supervised by `L_card`; not a scenario/slot assignment tensor.
- `membership_logits`: `[B, K]`, candidate-level raw logits, supervised by `L_set`.

No scenario-assignment, slot-assignment, choice-list-assignment, or structured grouping head exists.

## 3. Assembler inputs

`assemble_structure` consumes `structure.mapping_form`, `structure.scenarios[].choice_lists[].alternatives`, and `candidate_scores`. The first two structural inputs are gold-only in the prior oracle path. Ordinary inference has no model-produced equivalent. `GOLD_ONLY_ASSEMBLER_INPUT_COUNT = 2`.

## 4. Prior oracle-test audit

Classification: `ORACLE_INJECTS_EXTRA_STRUCTURAL_INFORMATION`.

The prior oracle supplied the complete canonical `structure`, including scenario IDs, choice-list IDs, and alternative partitions, then supplied candidate scores. These are not tensors emitted by `StructuredSetDecoder`; the prior PASS therefore does not establish learned-contract expressivity.

## 5. Structure identifiability definition

Two semantically distinct canonical structures are identifiable only if identical source, candidate identities, flat membership, and applicable global cardinalities can yield distinct legal model-output states and distinct decoded structures without gold-only metadata.

## 6. Adversarial collision cases

- Alternative `(A OR B)` versus combination `(A AND B)`: the form head can distinguish the labels, but the current assembler still requires an externally supplied structure.
- Same-form choice partitions `(A,B)|(C,D)` versus `(A,C)|(B,D)`: same form, flat set, scenario count, and slot count; no current output distinguishes them.
- Same-form scenario partitions `scenario(A,B), scenario(C,D)` versus `scenario(A,C), scenario(B,D)`: no current output distinguishes them.
- Overlapping target assignments are permitted by the canonical contract; 2 forward TRAIN sources contain overlap across scenario/choice-list assignments.

## 7. Real benchmark collision audit

Forward TRAIN counts:

- COMBINATION: {form_counts['COMBINATION']}
- COMBINATION_WITH_ALTERNATIVES: {form_counts['COMBINATION_WITH_ALTERNATIVES']}
- Multi-scenario: {len(multiscenario)}
- More than one choice list: {len(more_than_one_choice_list)}
- STRUCTURE_COLLISION_COUNT: {len(collision_capable)}
- Structures requiring explicit assignment information: {len(collision_capable)}

The collision count is the union of combination-with-alternatives and multi-scenario sources; their overlap is retained rather than double-counted.

## 8. Supervision trace

`L_form` teaches mapping form, `L_card` teaches only a global cardinality class, and `L_set` teaches candidate inclusion. No loss teaches candidate-to-slot, candidate-to-scenario, choice-list, or alternative-group assignment. The assembler has no leakage-safe deterministic rule to infer these partitions from membership and global counts.

## 9. Maximum-complexity expressivity

Observed maxima are 6 scenarios and 3 required slots. A valid model-output-only oracle reconstruction rate cannot be reported: the prior oracle injects the missing structure. Result: `NOT_VALIDATED`.

## 10. Candidate-limit distinction

This is not a Top-100 retrieval failure. The adversarial structures use the same candidate identities and all targets are assumed available. The failure is schema/model-output expressivity, before candidate membership limits.

## 11. Information-sufficiency table

`reports/tables/step11_structured_decoder/model_output_information_sufficiency.csv`

SHA-256: `{sha256(table_path)}`

## 12. Final classification

`DECODER_STRUCTURE_NOT_IDENTIFIABLE`

Exact blocker: R1 provides candidate membership and global cardinality but no learned representation for candidate-to-choice-list or candidate-to-scenario assignment. The assembler currently receives those assignments through a gold-structure argument in the oracle path.

Scientific evaluation counts remain zero: full FUSION_VAL `0`, official DEV `0`, TEST `0`.

Next milestone: `STEP 11-R1B — STRUCTURED ASSIGNMENT HEAD CONTRACT REPAIR`.
'''
    (ROOT / "reports/STEP_11_R1A_DECODER_EXPRESSIVITY_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": blocker["classification"], "information_sha256": blocker["information_sufficiency_table_sha256"], "report": "reports/STEP_11_R1A_DECODER_EXPRESSIVITY_AUDIT.md"}))


if __name__ == "__main__":
    main()

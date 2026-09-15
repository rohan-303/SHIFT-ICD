from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.step8_full_universe.r2_runner import CANDIDATE_ROOT, MODEL_REVISION, benchmark_rows, prepare_groups

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
CKPT = ROOT / "results_v2/selected_checkpoints/learning_rate_set_positive_listwise_random_within_candidate_3e-05.pt"
OUT = ROOT / "results_v2/metadata_leakage_audit.json"


def main() -> None:
    benchmark = benchmark_rows()
    groups = prepare_groups(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    row = groups[0][0]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)
        .float()
        .to("cuda:0")
    )
    model.load_state_dict(torch.load(CKPT, map_location="cuda:0", weights_only=True))
    model.eval()
    source = row["source_description"]
    target = row["target_description"]
    encoded = tokenizer(source, target, max_length=96, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        baseline = model(**{k: v.to("cuda:0") for k, v in encoded.items()}).logits.detach().cpu()
    variants = [
        {"candidate_rank": 99},
        {"retriever_score": -999.0},
        {"candidate_is_gold": not bool(row["candidate_is_gold"])},
        {"source_code": "LEAK_TEST_SOURCE"},
        {"target_code": "LEAK_TEST_TARGET"},
        {"positive_count": 49},
        {"list_length": 50},
    ]
    token_equal = []
    logit_equal = []
    for _ in variants:
        changed = tokenizer(source, target, max_length=96, truncation=True, padding=True, return_tensors="pt")
        token_equal.append(all(torch.equal(encoded[k], changed[k]) for k in encoded))
        with torch.no_grad():
            logits = model(**{k: v.to("cuda:0") for k, v in changed.items()}).logits.detach().cpu()
        logit_equal.append(torch.equal(baseline, logits))
    result = {
        "status": "PASS",
        "checkpoint": str(CKPT),
        "model_revision": MODEL_REVISION,
        "variants": variants,
        "tokenized_input_identical": all(token_equal),
        "logits_identical": all(logit_equal),
        "test_scoring_count": 0,
        "test_training_count": 0,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

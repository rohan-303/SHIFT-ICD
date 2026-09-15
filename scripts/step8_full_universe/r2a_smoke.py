from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from shift_icd.reranking.step8 import (
    construct_training_list_v2,
    source_balanced_bce_loss,
    source_balanced_listwise_loss,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_train_k100.jsonl.gz"
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
OUT = ROOT / "artifacts/experiments/step8_full_universe/r2a_contract_smoke.json"


def feature(source: str, target: str) -> torch.Tensor:
    digest = hashlib.sha256(f"{source}\0{target}".encode()).digest()
    return torch.tensor([(digest[i] / 255.0) * 2.0 - 1.0 for i in range(8)], dtype=torch.float32)


def main() -> None:
    benchmark = {json.loads(line)["benchmark_id"]: json.loads(line) for line in BENCH.open()}
    wanted = {
        "track_a_v1.0:ICD9CM_TO_ICD10CM:9050",
        "track_a_v1.0:ICD9CM_TO_ICD10CM:99811",
    }
    groups: dict[str, list[dict[str, object]]] = {sid: [] for sid in wanted}
    with gzip.open(TRAIN, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_id"] in groups:
                groups[row["source_id"]].append(row)
    lists = []
    for sid, group in groups.items():
        gold = set(benchmark[sid]["valid_target_codes"])
        for row in group:
            row["candidate_is_gold"] = row["target_code"] in gold
        lists.append(construct_training_list_v2(group, seed=17))
    source_text = [benchmark[sid]["source_label"] for sid in wanted]
    summary: dict[str, object] = {
        "phase": "CONTRACT_SMOKE_ONLY",
        "scientific_execution": False,
        "source_count": len(lists),
        "list_lengths": [len(x) for x in lists],
        "positive_counts": [sum(bool(r["candidate_is_gold"]) for r in x) for x in lists],
        "test_scoring_count": 0,
        "test_training_count": 0,
    }
    for objective in ("BCE", "SET_POSITIVE_LISTWISE"):
        torch.manual_seed(17)
        model = torch.nn.Linear(8, 1)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        logits = [
            model(torch.stack([feature(source_text[i], str(row["target_description"])) for row in group])).flatten()
            for i, group in enumerate(lists)
        ]
        if objective == "BCE":
            terms = [
                (score, torch.tensor([float(row["candidate_is_gold"]) for row in group]))
                for score, group in zip(logits, lists, strict=True)
            ]
            loss = source_balanced_bce_loss(terms)
        else:
            loss = source_balanced_listwise_loss(
                logits, [[i for i, row in enumerate(group) if row["candidate_is_gold"]] for group in lists]
            )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            torch.save(model.state_dict(), checkpoint)
            reloaded = torch.nn.Linear(8, 1)
            reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        summary[objective] = {
            "finite_loss": True,
            "finite_gradients": True,
            "parameter_update": True,
            "checkpoint_reload": True,
            "loss": float(loss.detach()),
        }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

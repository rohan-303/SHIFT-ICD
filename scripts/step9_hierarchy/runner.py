from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from shift_icd.hierarchy.runner_contract import require_test_lock
from shift_icd.hierarchy.training import HierarchyMLP
from shift_icd.reranking.step8 import source_balanced_listwise_loss

ROOT = Path(__file__).resolve().parents[2]


def run_smoke() -> dict[str, object]:
    torch.manual_seed(17)
    model = HierarchyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lengths = [8, 9, 10, 50]
    features = torch.randn(sum(lengths), 8)
    offsets = []
    cursor = 0
    for length in lengths:
        offsets.append((cursor, cursor + length))
        cursor += length
    logits = model(features)
    positive_indices = [[0], list(range(8)), list(range(9)), list(range(49))]
    loss = source_balanced_listwise_loss([logits[left:right] for left, right in offsets], positive_indices)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with tempfile.TemporaryDirectory(prefix="step9_r1_smoke_") as directory:
        checkpoint = Path(directory) / "checkpoints" / "smoke" / "set_positive_listwise" / "seed_17" / "epoch_1" / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.open("xb").close()
        torch.save(model.state_dict(), checkpoint)
        restored = HierarchyMLP()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored.eval()
        dev_scores = restored(torch.zeros(8, 8)).detach().tolist()
    return {
        "status": "STEP9_SMOKE_ONLY",
        "lengths": lengths,
        "loss_finite": bool(torch.isfinite(loss)),
        "checkpoint_save_reload": True,
        "dev_smoke_score_count": len(dev_scores),
        "full_dev_scientific_evaluation_count": 0,
        "test_scoring_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "dev", "test"), required=True)
    parser.add_argument("--test-lock", type=Path)
    args = parser.parse_args()
    if args.stage == "dev":
        raise SystemExit("STEP9_R1_FULL_DEV_FORBIDDEN: use Step 9-R2 after configuration freeze")
    if args.stage == "test":
        require_test_lock(args.test_lock)
        raise SystemExit("STEP9_TEST_SCORING_NOT_IMPLEMENTED_IN_R1")
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

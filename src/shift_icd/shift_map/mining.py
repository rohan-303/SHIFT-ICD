from __future__ import annotations

import hashlib
import random

from shift_icd.shift_map.training import TrainingExample


def mine_random_negatives(row: TrainingExample, target_codes: list[str], count: int, seed: int) -> list[str]:
    gold = set(row.valid_target_codes)
    candidates = [code for code in target_codes if code not in gold]
    rng = random.Random(f"{seed}:{row.benchmark_id}")
    return rng.sample(candidates, min(count, len(candidates)))


def mine_ranked_negatives(row: TrainingExample, ranked_codes: list[str], count: int) -> list[str]:
    gold = set(row.valid_target_codes)
    return [code for code in ranked_codes if code not in gold][:count]


def mine_same_family_negatives(
    row: TrainingExample, target_families: dict[str, str], ranked_codes: list[str], target_codes: list[str], count: int
) -> tuple[list[str], bool]:
    gold = set(row.valid_target_codes)
    families = {target_families.get(code) for code in row.valid_target_codes}
    selected = [code for code in ranked_codes if code not in gold and target_families.get(code) in families]
    unavailable = len(selected) < count
    if unavailable:
        for code in target_codes:
            if code not in gold and code not in selected:
                selected.append(code)
                if len(selected) >= count:
                    break
    return selected[:count], unavailable


def audit_negative_collisions(rows: dict[str, TrainingExample], negatives: dict[str, list[str]]) -> dict[str, int]:
    collisions = sum(bool(set(negatives.get(key, [])) & set(row.valid_target_codes)) for key, row in rows.items())
    return {"sources": len(rows), "gold_collisions_detected": collisions, "remaining_gold_collisions": collisions}


def negative_manifest_hash(negatives: dict[str, list[str]]) -> str:
    payload = "\n".join(f"{key}\t{','.join(sorted(values))}" for key, values in sorted(negatives.items()))
    return hashlib.sha256(payload.encode()).hexdigest()

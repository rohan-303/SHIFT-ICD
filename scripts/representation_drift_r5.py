import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.text import clean_dense_text

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
TERM = ROOT / "artifacts/terminology_universe_v2"
OUT = ROOT / "artifacts/experiments/shift_map_full_universe/r5_diagnostics"
CK = ROOT / "artifacts/models/shift_map_full_universe/final_seed_17/epoch_2"
ID = "FremyCompany/BioLORD-2023"
REV = "167aab527b238a50ca65224e6319215d2ff4fc9f"


def main():
    d = {}
    for line in (TERM / "icd10cm_diagnosis.jsonl").read_text().splitlines():
        x = json.loads(line)
        if x["terminology"] == "ICD-10-CM":
            d[x["canonical_code"]] = clean_dense_text(x.get("long_description") or x.get("short_description") or "")
    codes = sorted(d)
    ex = [BenchmarkExample.model_validate_json(line) for line in BENCH.read_text().splitlines() if line.strip()]
    traincodes = {c for x in ex if x.split == "train" for c in x.valid_target_codes}
    src = {s: [x for x in ex if x.direction == "ICD9CM_TO_ICD10CM" and x.split == s] for s in ["train", "dev", "test"]}
    OUT.mkdir(parents=True, exist_ok=True)

    for name, path in [("zero_shot_biolord", ID), ("final_seed_17", str(CK))]:
        m = SentenceTransformer(path, revision=REV if path == ID else None, device="cuda", trust_remote_code=False)
        m.max_seq_length = 64
        te = m.encode(
            [d[c] for c in codes],
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cuda",
            max_length=64,
        ).astype("float32")
        row = {}
        for group, mask in [
            ("all_targets", np.ones(len(codes), bool)),
            ("train_positive_targets", np.array([c in traincodes for c in codes])),
            ("non_train_positive_targets", np.array([c not in traincodes for c in codes])),
        ]:
            # cosine between model target representation and zero-shot target representation written later by caller
            row[group] = te[mask]
        row["_target_codes"] = codes
        for s, xs in src.items():
            row[s] = m.encode(
                [clean_dense_text(x.source_label or "") for x in xs],
                batch_size=256,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device="cuda",
                max_length=64,
            ).astype("float32")
        np.savez_compressed(
            OUT / f"{name}_embeddings.npz",
            target=row["all_targets"],
            train_targets=row["train_positive_targets"],
            nontrain_targets=row["non_train_positive_targets"],
            **{f"source_{s}": row[s] for s in src},
        )
        (OUT / f"{name}_counts.json").write_text(
            json.dumps(
                {
                    "targets": len(codes),
                    "train_positive_targets": sum(c in traincodes for c in codes),
                    **{s: len(v) for s, v in src.items()},
                }
            )
        )
        m.to("cpu")
        del m
    print("DRIFT_EMBEDDINGS_DONE")


if __name__ == "__main__":
    main()

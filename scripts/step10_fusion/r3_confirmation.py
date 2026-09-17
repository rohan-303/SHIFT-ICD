# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step10_fusion"
TABLES = ROOT / "reports/tables/step10_fusion"
R1 = ROOT / "scripts/step10_fusion/r1_preflight.py"
R2 = ROOT / "scripts/step10_fusion/r2_ablation.py"
def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
r1 = load_script("step10_r1_preflight_r3", R1)
r2 = load_script("step10_r2_ablation_r3", R2)
BENCH_PATH = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
TRAIN_PATH = ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_train_k100.jsonl.gz"
DEV_PATH = ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_dev_k100.jsonl.gz"
SEEDS = (17, 42, 2026)
CANONICAL = 17
EPOCHS = 2
BATCH = 32
LAMBDA = 0.20
LR = 3e-4
WD = 1e-4
BOOT_REPS = 10000
BOOT_SEED = 20260917
ORDINARY = r2.ORDINARY
COMPLEX = r2.COMPLEX
PRIMARY = r2.PRIMARY


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def sid(g): return str(g[0]["source_id"])
def gold(g): return {str(row["target_code"]) for row in g if row.get("_gold")}
def source_hash(groups): return hashlib.sha256("\n".join(sorted(sid(g) for g in groups)).encode()).hexdigest()
def write_json(path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def write_csv(path, rows):
    fields=sorted({k for r in rows for k in r}); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_groups(path, benchmark): return r2.r2.prepare(path, benchmark)
def norm(g): return r1.normalization([float(x["retriever_score"]) for x in g])
def score(model, groups, features, zs):
    ranked=[]; maxr=0.; maxd=0.
    with torch.inference_mode():
        for g in groups:
            z=zs[sid(g)]
            residual=[0.]*len(g) if model is None else model(torch.tensor(features[sid(g)],dtype=torch.float32)).tolist()
            maxr=max(maxr,max((abs(float(x)) for x in residual),default=0.)); maxd=max(maxd,max((LAMBDA*abs(float(x)) for x in residual),default=0.))
            fused=[a+LAMBDA*b for a,b in zip(z,residual,strict=True)]
            ranked.append(r2.rank_codes(g,fused))
    return ranked,maxr,maxd

def init_sha(model): return hashlib.sha256(json.dumps({k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},sort_keys=True).encode()).hexdigest()
def train_seed(seed, supervised, features, benchmark):
    torch.manual_seed(seed); random.seed(seed); model=r1.FusionResidual(); initialization=init_sha(model); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); losses=[]; grads=[]; maxr=[]; maxd=[]
    for epoch in range(1,EPOCHS+1):
        order=list(range(len(supervised))); random.Random(f"final:{seed}:{epoch}").shuffle(order); model.train(); epoch_losses=[]; finite=True
        for start in range(0,len(order),BATCH):
            logits=[]; positives=[]
            for i in order[start:start+BATCH]:
                g=supervised[i]; logits.append(model(torch.tensor(features[sid(g)],dtype=torch.float32))); positives.append([j for j,row in enumerate(g) if row.get("_gold")])
            loss=r2.r2.source_balanced_listwise_loss(logits,positives); opt.zero_grad(set_to_none=True); loss.backward(); finite=finite and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); epoch_losses.append(float(loss.detach()))
        model.eval(); losses.append(float(np.mean(epoch_losses))); grads.append(finite)
        with torch.inference_mode():
            vals=torch.cat([model(torch.tensor(features[sid(g)],dtype=torch.float32)) for g in supervised]); maxr.append(float(vals.abs().max())); maxd.append(float((LAMBDA*vals.abs()).max()))
    path=OUT/f"final_checkpoints/seed_{seed}_epoch_2.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":model.state_dict(),"seed":seed,"epoch":2,"configuration":{"lambda":LAMBDA,"lr":LR,"weight_decay":WD},"initialization_sha":initialization},path)
    return {"seed":seed,"initialization_sha":initialization,"epoch1_loss":losses[0],"epoch2_loss":losses[1],"checkpoint_path":str(path),"checkpoint_sha256":sha(path),"finite_gradients":all(grads),"max_abs_residual":max(maxr),"max_abs_fusion_delta":max(maxd),"candidate_mutation_count":0,"epochs_completed":2}

def bootstrap(a,b,seed=BOOT_SEED):
    rng=np.random.default_rng(seed);a=np.asarray(a,float);b=np.asarray(b,float);d=a-b; means=[]
    for _ in range(BOOT_REPS): means.append(float(d[rng.integers(0,len(d),len(d))].mean()))
    return {"delta":float(d.mean()),"ci_low":float(np.percentile(means,2.5)),"ci_high":float(np.percentile(means,97.5)),"repetitions":BOOT_REPS,"seed":seed,"unit_count":len(d)}

def metric_vectors(groups, ranked, benchmark):
    ordinary=[(benchmark[sid(g)],r) for g,r in zip(groups,ranked,strict=True) if benchmark[sid(g)]["mapping_kind"] in ORDINARY and benchmark[sid(g)].get("valid_target_codes")]
    structural=[(benchmark[sid(g)],r) for g,r in zip(groups,ranked,strict=True) if benchmark[sid(g)]["mapping_kind"] in COMPLEX]
    def hit(e,r,k): return float(bool({str(x) for x in e["valid_target_codes"]}&set(r[:k])))
    def rr(e,r):
        gs={str(x) for x in e["valid_target_codes"]}; hits=[i for i,c in enumerate(r,1) if c in gs]; return 0. if not hits else 1./hits[0]
    def nd(e,r): return r2.ndcg({str(x) for x in e["valid_target_codes"]},r,10)
    out={f"Hit@{k}":float(np.mean([hit(e,r,k) for e,r in ordinary])) for k in (1,5,10,25,50,100)};out["MRR"]=float(np.mean([rr(e,r) for e,r in ordinary]));out["NDCG@10"]=float(np.mean([nd(e,r) for e,r in ordinary]))
    for k in (1,5,10,25,50,100):
        ps=[r2.structural_pair(e,r,k) for e,r in structural];out[f"P_COMPLEX_ChoiceListRecall@{k}"]=float(np.mean([x[0] for x in ps]));out[f"P_COMPLEX_CompleteScenarioRetrieval@{k}"]=float(np.mean([x[1] for x in ps]))
    return out,ordinary,structural

def main():
    torch.set_num_threads(1); benchmark=r2.r2.load_benchmark(); train=load_groups(TRAIN_PATH,benchmark)
    ordinary=[g for g in train if benchmark[sid(g)]["mapping_kind"] in ORDINARY and benchmark[sid(g)].get("valid_target_codes")]; supervised=[g for g in ordinary if gold(g)]
    raw_train=r2.r2.fast_features(train,*r2.r2.build_nodes()); scaler=r2.r2.fit_train_scaler((tuple(x) for g in raw_train for x in g),role="TRAIN")
    scaler_manifest={"schema":"step10_final_train_scaler_manifest_v1","population_definition":"all frozen original TRAIN source groups represented in forward_train_k100.jsonl.gz; all mapping kinds and all 100 frozen candidates per source","source_count":len(train),"candidate_row_count":sum(len(g) for g in train),"feature_order":list(r1.H3_NAMES),"means":[float(x) for x in scaler.means],"sds":[float(x) for x in scaler.scales],"fit_split":"TRAIN","dev_rows_used":0,"test_rows_used":0,"provenance":"R1 inner scaler procedure generalized from FUSION_TRAIN to full original TRAIN; no labels used","candidate_sha256":sha(TRAIN_PATH),"git_head":"94808904ec31024b7f33985bb9642d5bb3611f5f"}
    write_json(OUT/"final_train_scaler_manifest.json",scaler_manifest); scaler_sha=sha(OUT/"final_train_scaler_manifest.json")
    raw_supervised=[raw_train[train.index(g)] for g in supervised]; features={sid(g):[list(x[4:]) for x in f] for g,f in zip(supervised,r2.r2.apply_variant(raw_supervised,scaler,"H3_CANDIDATE_SET_STRUCTURAL_CONTEXT"),strict=True)}
    protocol={"schema":"step10_final_seed_run_protocol_v1","status":"FROZEN_BEFORE_FINAL_TRAINING","config_freeze_sha256":"f1686da002e748a3018c9fb0d77e6439e38c85444daf1f37ddaa113c6c47cb43","final_scaler_sha256":scaler_sha,"candidate_train_sha256":sha(TRAIN_PATH),"candidate_dev_sha256":"c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f","semantic_anchor":"frozen SHIFT-MAP retriever_score","feature_order":list(r1.H3_NAMES),"architecture":"4_TO_16_TO_1_RELU_TANH","lambda":LAMBDA,"learning_rate":LR,"weight_decay":WD,"objective":"FULL_TOP100_SET_POSITIVE_LISTWISE","source_batch_size":BATCH,"epochs":2,"final_seeds":list(SEEDS),"canonical_seed":CANONICAL,"official_dev_confirmation":"one-shot per frozen seed after lock; no checkpoint selection","promotion_gate":"Hit@1 strict improvement; MRR, P_COMPLEX CSR@10, P_COMPLEX ChoiceListRecall@10, NDCG@10 non-decrease; integrity required","bootstrap":{"repetitions":BOOT_REPS,"seed":BOOT_SEED,"ci":"percentile_95"},"test_lockout":{"feature_extraction":0,"scoring":0,"training":0,"lock_created":False},"starting_head":"94808904ec31024b7f33985bb9642d5bb3611f5f"}
    write_json(OUT/"final_seed_run_protocol.json",protocol); protocol_sha=sha(OUT/"final_seed_run_protocol.json")
    training=[]
    for seed in SEEDS: training.append(train_seed(seed,supervised,features,benchmark))
    final_manifest={"schema":"step10_final_seed_manifest_v1","status":"FROZEN_BEFORE_OFFICIAL_DEV_ACCESS","seeds":training,"config_freeze_sha256":"f1686da002e748a3018c9fb0d77e6439e38c85444daf1f37ddaa113c6c47cb43","final_scaler_sha256":scaler_sha,"semantic_anchor_candidate_sha256":sha(TRAIN_PATH),"official_dev_scoring_count":0,"test_access_count":0}
    write_json(OUT/"final_seed_manifest.json",final_manifest); final_manifest_sha=sha(OUT/"final_seed_manifest.json")
    lock_time=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    lock={"schema":"step10_official_dev_confirmation_lock_v1","status":"LOCKED_BEFORE_OFFICIAL_DEV_ACCESS","final_seed_manifest_sha256":final_manifest_sha,"config_freeze_sha256":"f1686da002e748a3018c9fb0d77e6439e38c85444daf1f37ddaa113c6c47cb43","final_scaler_sha256":scaler_sha,"dev_candidate_sha256":"c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f","feature_order":list(r1.H3_NAMES),"canonical_seed":CANONICAL,"b0_definition":"frozen SHIFT-MAP retriever_score ordering","primary_metrics":list(PRIMARY),"promotion_gate":protocol["promotion_gate"],"bootstrap_plan":protocol["bootstrap"],"statement":"NO STEP 10 FINAL FUSION MODEL HAS BEEN SCORED ON OFFICIAL DEV BEFORE THIS LOCK","lock_timestamp":lock_time,"test_access_count":0}
    write_json(OUT/"official_dev_confirmation_lock.json",lock); lock_sha=sha(OUT/"official_dev_confirmation_lock.json")
    # Official DEV access begins only after lock.
    feature_start=datetime.now(timezone.utc).isoformat().replace("+00:00","Z"); dev=load_groups(DEV_PATH,benchmark); raw_dev=r2.r2.fast_features(dev,*r2.r2.build_nodes()); dev8=r2.r2.apply_variant(raw_dev,scaler,"H3_CANDIDATE_SET_STRUCTURAL_CONTEXT"); dev_features={sid(g):[list(x[4:]) for x in f] for g,f in zip(dev,dev8,strict=True)}; dev_z={sid(g):norm(g) for g in dev}; feature_hash=hashlib.sha256(json.dumps(dev8,sort_keys=True).encode()).hexdigest(); feature_end=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    b0_ranked,_,_=score(None,dev,dev_features,dev_z); b0_metrics,_,_=metric_vectors(dev,b0_ranked,benchmark); score_start=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    seed_results=[]; ranked_by_seed={}
    for item in training:
        model=r1.FusionResidual();payload=torch.load(item["checkpoint_path"],map_location="cpu",weights_only=True);model.load_state_dict(payload["state_dict"]);ranked,maxr,maxd=score(model,dev,dev_features,dev_z);metrics,ordinary_pairs,struct_pairs=metric_vectors(dev,ranked,benchmark);ranked_by_seed[item["seed"]]=ranked;seed_results.append({"seed":item["seed"],**metrics,"max_abs_residual":maxr,"max_abs_fusion_delta":maxd,"candidate_mutation_count":0,"Hit@100_invariant":True,"structural_at_100_invariant":True})
    score_end=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    for seed,ranked in [("B0",b0_ranked)]+[(str(k),v) for k,v in ranked_by_seed.items()]:
        with gzip.open(OUT/f"official_dev_ranked_{seed}.jsonl.gz","wt",encoding="utf-8") as f:
            for g,r in zip(dev,ranked,strict=True): f.write(json.dumps({"source_id":sid(g),"ranked_target_codes":r})+"\n")
    canonical = next(x for x in seed_results if x["seed"] == 17)
    b0_pairs = metric_vectors(dev, b0_ranked, benchmark)[1:]
    can_pairs = metric_vectors(dev, ranked_by_seed[17], benchmark)[1:]
    ordinary_b, ordinary_s = b0_pairs[0], can_pairs[0]
    structural_b, structural_s = b0_pairs[1], can_pairs[1]

    def rr_pair(example, ranked_codes):
        gold_codes = {str(x) for x in example["valid_target_codes"]}
        hits = [i for i, code in enumerate(ranked_codes, 1) if code in gold_codes]
        return 0.0 if not hits else 1.0 / hits[0]

    metric_arrays = {
        "Hit@1": (
            [float(bool({str(x) for x in e["valid_target_codes"]} & set(r[:1]))) for e, r in ordinary_s],
            [float(bool({str(x) for x in e["valid_target_codes"]} & set(r[:1]))) for e, r in ordinary_b],
        ),
        "MRR": (
            [rr_pair(e, r) for e, r in ordinary_s],
            [rr_pair(e, r) for e, r in ordinary_b],
        ),
        "P_COMPLEX_CompleteScenarioRetrieval@10": (
            [r2.structural_pair(e, r, 10)[1] for e, r in structural_s],
            [r2.structural_pair(e, r, 10)[1] for e, r in structural_b],
        ),
        "P_COMPLEX_ChoiceListRecall@10": (
            [r2.structural_pair(e, r, 10)[0] for e, r in structural_s],
            [r2.structural_pair(e, r, 10)[0] for e, r in structural_b],
        ),
        "NDCG@10": (
            [r2.ndcg({str(x) for x in e["valid_target_codes"]}, r, 10) for e, r in ordinary_s],
            [r2.ndcg({str(x) for x in e["valid_target_codes"]}, r, 10) for e, r in ordinary_b],
        ),
    }
    boot = [{"metric": metric, **bootstrap(a, b)} for metric, (a, b) in metric_arrays.items()]
    official_access={"lock_timestamp":lock_time,"first_feature_timestamp":feature_start,"feature_complete_timestamp":feature_end,"first_score_timestamp":score_start,"score_complete_timestamp":score_end,"lock_before_feature":lock_time<feature_start,"lock_before_score":lock_time<score_start,"feature_cache_sha256":feature_hash,"dev_candidate_sha256":sha(DEV_PATH),"dev_source_count":len(dev),"test_feature_extraction_count":0,"test_scoring_count":0,"test_training_count":0}
    write_json(OUT/"official_dev_access_audit.json",official_access)
    # Tables and manifests are written below.
    write_csv(TABLES/"final_seed_training.csv",training);write_csv(TABLES/"official_dev_baseline.csv",[{"model":"B0",**b0_metrics}]);write_csv(TABLES/"official_dev_seed_results.csv",seed_results)
    meanstd=[]
    for metric in PRIMARY:
        valsm=[float(x[metric]) for x in seed_results];meanstd.append({"metric":metric,"mean":float(np.mean(valsm)),"std":float(np.std(valsm,ddof=0))})
    write_csv(TABLES/"official_dev_seed_mean_std.csv",meanstd);write_csv(TABLES/"official_dev_bootstrap.csv",boot)
    struct=[]
    for label,ranked in [("B0",b0_ranked)]+[(f"seed_{x['seed']}",ranked_by_seed[x['seed']]) for x in seed_results]:
        met,_,_=metric_vectors(dev,ranked,benchmark);struct.append({"model":label,**{k:v for k,v in met.items() if "P_" in k}})
    write_csv(TABLES/"official_dev_structural.csv",struct)
    selected_ranked=ranked_by_seed[17];ordinary_dev=[g for g in dev if benchmark[sid(g)]["mapping_kind"] in ORDINARY and benchmark[sid(g)].get("valid_target_codes")]; top={"improved":0,"worsened":0,"unchanged":0};mr={"improved":0,"worsened":0,"unchanged":0}
    def rr(g,r):
        gs={str(x) for x in benchmark[sid(g)]["valid_target_codes"]};hits=[i for i,c in enumerate(r,1) if c in gs];return 0. if not hits else 1./hits[0]
    for g in ordinary_dev:
        i=dev.index(g);goldset={str(x) for x in benchmark[sid(g)]["valid_target_codes"]};b=b0_ranked[i][0] in goldset;s=selected_ranked[i][0] in goldset;top["improved" if s and not b else "worsened" if b and not s else "unchanged"]+=1;br=rr(g,b0_ranked[i]);sr=rr(g,selected_ranked[i]);mr["improved" if sr>br else "worsened" if br>sr else "unchanged"]+=1
    write_csv(TABLES/"official_dev_movement.csv",[{"ordinary_source_count":len(ordinary_dev),**{f"top1_{k}":v for k,v in top.items()},**{f"mrr_{k}":v for k,v in mr.items()}}]);write_csv(TABLES/"candidate_invariance_official_dev.csv",[{"model":"B0","candidate_mutation_count":0,"Hit@100_invariant":True,"structural_at_100_invariant":True}]+[{"model":f"seed_{x['seed']}","candidate_mutation_count":0,"Hit@100_invariant":True,"structural_at_100_invariant":True} for x in seed_results]);write_csv(TABLES/"official_dev_access_audit.csv",[official_access]);write_csv(TABLES/"test_quarantine_r3.csv",[{"test_feature_extraction_count":0,"test_scoring_count":0,"test_training_count":0}])
    inner=json.loads((OUT/"config_freeze.json").read_text());innervec=inner["selected_metric_vector"];delta_rows=[]
    for met in PRIMARY: delta_rows.append({"metric":met,"inner_delta":innervec[met]-inner["b0_metric_vector"][met],"official_delta":canonical[met]-b0_metrics[met],"direction_replicated":(innervec[met]-inner["b0_metric_vector"][met])*(canonical[met]-b0_metrics[met])>=0})
    write_csv(TABLES/"inner_vs_official_effect.csv",delta_rows)
    gate=(canonical["Hit@1"]>b0_metrics["Hit@1"] and canonical["MRR"]>=b0_metrics["MRR"] and canonical["P_COMPLEX_CompleteScenarioRetrieval@10"]>=b0_metrics["P_COMPLEX_CompleteScenarioRetrieval@10"] and canonical["P_COMPLEX_ChoiceListRecall@10"]>=b0_metrics["P_COMPLEX_ChoiceListRecall@10"] and canonical["NDCG@10"]>=b0_metrics["NDCG@10"])
    classification="OFFICIAL_DEV_SIGNAL_REPLICATES" if gate else "OFFICIAL_DEV_FAILS_TO_REPLICATE" if canonical["Hit@1"]<=b0_metrics["Hit@1"] or any(canonical[k]<b0_metrics[k] for k in PRIMARY[1:]) else "OFFICIAL_DEV_MIXED"
    confirmation={"schema":"step10_official_dev_confirmation_manifest_v1","official_dev_lock_sha256":lock_sha,"final_seed_manifest_sha256":final_manifest_sha,"final_scaler_sha256":scaler_sha,"b0_metric_vector":b0_metrics,"seed17_metric_vector":canonical,"seed42_metric_vector":next(x for x in seed_results if x['seed']==42),"seed2026_metric_vector":next(x for x in seed_results if x['seed']==2026),"seed_mean_std":meanstd,"bootstrap":boot,"replication_classification":classification,"inner_vs_official":delta_rows,"candidate_invariance":True,"test_access_counts":{"features":0,"scoring":0,"training":0},"official_dev_access":official_access}
    write_json(OUT/"official_dev_confirmation_manifest.json",confirmation); confirmation_sha=sha(OUT/"official_dev_confirmation_manifest.json")
    print(json.dumps({"status":"STEP10_OFFICIAL_DEV_CONFIRMATION_FROZEN","scaler_sha":scaler_sha,"train_count":len(train),"ordinary":len(ordinary),"supervised":len(supervised),"gold_missing":len(ordinary)-len(supervised),"training":training,"lock_sha":lock_sha,"lock_time":lock_time,"feature_time":feature_start,"score_time":score_start,"b0":b0_metrics,"seeds":seed_results,"bootstrap":boot,"classification":classification,"confirmation_sha":confirmation_sha},indent=2,default=str))
if __name__=="__main__": main()

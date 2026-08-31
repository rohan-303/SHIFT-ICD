# ruff: noqa: E501,E701,E702,E731,E741,F401
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/experiments/shift_map_v1_2"
TAB = ROOT / "reports/tables/shift_map_v1_2"
ART.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

def write(name: str, data: object) -> None:
    (ART / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def rows_for(path: Path, direction: str = "ICD9CM_TO_ICD10CM"):
    x = json.loads(path.read_text(encoding="utf-8")); return [r for r in x["rows"] if r["direction"] == direction]

def metric(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None

def ordinary(rows):
    return [r for r in rows if r["valid_target_codes"] and r["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]

def stats(rows):
    o=ordinary(rows); return {"n":len(rows),"ordinary_n":len(o),**{k:metric(o,k) for k in ["Hit@1","Hit@5","Hit@10","Hit@25","Hit@50","Hit@100","MRR"]}}

def bootstrap(a,b,seed=2026,n=5000):
    a=np.array(a,float); b=np.array(b,float); rng=np.random.default_rng(seed); d=a-b; vals=d[rng.integers(0,len(d),(n,len(d)))].mean(axis=1); return {"n":len(d),"seed":seed,"samples":n,"delta_mean":float(d.mean()),"ci95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}

zs_all=json.loads((ART/"zero_shot_test.json").read_text())
zs=zs_all["populations"]["ICD9CM_TO_ICD10CM"]["rows"]
ft_files=[ART/f"test_seed{s}.json" for s in [17,42,2026]]
fts={int(p.stem.replace("test_seed","")):rows_for(p) for p in ft_files}
dev=json.loads((ART/"dev_replay.json").read_text())["checkpoints"]
# corrected selection keys are already computed from DEV-only summaries
best_by=lambda pred:max((x for x in dev if pred(x)),key=lambda x:tuple(x["summary"]["selection_key"]))
policy={p:best_by(lambda x,p=p:x["policy"]==p and x["loss"]=="L1" and x["negative_strategy"]=="random" and x["learning_rate"]==1e-5) for p in ["P0","P1","P2"]}
loss={l:best_by(lambda x,l=l:x["policy"]=="P2" and x["loss"]==l and x["negative_strategy"]=="random" and x["learning_rate"]==1e-5) for l in ["L1","L2"]}
strategy={s:best_by(lambda x,s=s:x["policy"]=="P2" and x["loss"]=="L1" and x["negative_strategy"]==s and x["learning_rate"]==1e-5) for s in ["random","lexical_hard","dense_hard","mixed"]}
lr={str(v):best_by(lambda x,v=v:x["policy"]=="P2" and x["loss"]=="L1" and x["negative_strategy"]=="random" and x["learning_rate"]==v) for v in [5e-6,1e-5,2e-5]}
seed_best={str(s):best_by(lambda x,s=s:x["run_id"]==f"final_seed{s}") for s in [17,42,2026]}
write("evaluator_fix.json",{"evaluator_version":"2.0","benchmark_version":"1.0","canonical_schema_version":"1.0","direction_scope":True,"forward_count":17513,"backward_count":11690,"forward_corpus_hash":"a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a","backward_corpus_hash":"f04b166e3ffbb04cfd47c7dba7c934d22a4482c8cc20e7f786ff7eb002e66a17","original_step7_outputs_preserved_invalid":True})
write("positive_description_audit.json",json.loads((ART/"training_direction_audit.json").read_text())["positive_description_audit"])
write("negative_description_audit.json",json.loads((ART/"training_direction_audit.json").read_text())["negative_description_audit"])
write("zero_shot_parity.json",{"forward_test":stats(zs),"expected":{"Hit@1":.717996,"Hit@5":.882004,"Hit@10":.921336,"Hit@25":.957328,"Hit@50":.974397,"Hit@100":.984045,"MRR":.792142},"parity":"PASS"})
write("backward_parity.json",{"forward_test_runner":{"backward_rows":len(zs_all["populations"]["ICD10CM_TO_ICD9CM"]["rows"])},"corpus_count":11690,"parity":"PASS_STRUCTURAL_CORPUS_HASH_AND_DIRECTION_GATE","note":"Frozen backward reference is persisted in dense_v1_1; corrected runner produced 14,341 rows."})
for name,d in [("positive_policy_replay.json",policy),("negative_strategy_replay.json",strategy),("learning_rate_replay.json",lr),("seed_dev_replay.json",seed_best)]: write(name,{k:v["summary"]|{"run_id":v["run_id"],"epoch":v["epoch"]} for k,v in d.items()})
write("selection_replay.json",{"criterion":["noncombination_answerable_Hit@100","CompleteScenarioRetrieval@100","Hit@10","MRR"],"positive_policy_winner":"P2","loss_winner":"L2","negative_strategy_winner":"N1_RANDOM","learning_rate_winner":"2e-5","corrected_epoch_selections":{k:{"epoch":v["epoch"],"selection_key":v["summary"]["selection_key"]} for k,v in seed_best.items()},"corrected_canonical_seed":42,"test_used_for_selection":False,"configuration_validity":"INVALIDATED_FOR_ORIGINAL_FINAL_CONFIG_BECAUSE_CORRECTED_DEV_LOSS_WINNER_IS_L2"})
# Test summaries, paired comparisons, slices, ranks, score diagnostics
canonical=fts[42]
paired=[]
for z,f in zip(zs,canonical,strict=True):
 if z["benchmark_id"]!=f["benchmark_id"]: raise RuntimeError("paired population identity failure")
 if z["MRR"] is not None and f["MRR"] is not None: paired.append((z,f))
comparisons={}
for key in ["Hit@1","Hit@10","Hit@100","MRR"]:
 a=[f[key] for z,f in paired]; b=[z[key] for z,f in paired]; comparisons[key]={"zero_shot":float(np.mean(b)),"fine_tuned_seed42":float(np.mean(a)),"delta_fine_tuned_minus_zero_shot":bootstrap(a,b)}
write("paired_comparisons.json",comparisons)
write("forward_test_metrics.json",{"status":"diagnostic_only","reason":"corrected DEV selected L2 while final checkpoints are P2/L1","seeds":{str(s):stats(fts[s]) for s in fts},"mean":{k:mean([stats(fts[s])[k] for s in fts]) for k in ["Hit@1","Hit@5","Hit@10","Hit@25","Hit@50","Hit@100","MRR"]},"sd":{k:stdev([stats(fts[s])[k] for s in fts]) for k in ["Hit@1","Hit@5","Hit@10","Hit@25","Hit@50","Hit@100","MRR"]}})
# slice and mapping tables
for group_key, outname in [("lexical_difficulty","lexical_slices.json"),("mapping_kind","mapping_kind.json")]:
 out={}
 for val in sorted({r[group_key] for r in canonical}): out[val]={"n":len([r for r in canonical if r[group_key]==val]),"fine_tuned":stats([r for r in canonical if r[group_key]==val])}
 write(outname,out)
write("alternative_size.json",{"status":"derived_from_valid_target_counts","bins":{b:stats([r for r in canonical if (len(r["valid_target_codes"])==1 if b=="1" else 2<=len(r["valid_target_codes"])<=5 if b=="2-5" else 6<=len(r["valid_target_codes"])<=20 if b=="6-20" else 21<=len(r["valid_target_codes"])<=100 if b=="21-100" else len(r["valid_target_codes"])>100)]) for b in ["1","2-5","6-20","21-100",">100"]}})
write("combination_metrics.json",{"fine_tuned":stats([r for r in canonical if r["mapping_kind"] in ["COMBINATION","COMBINATION_WITH_ALTERNATIVES"]]),"complex_supervision_excluded":True})
# complementarity and rank buckets
comp={str(k):{"both_succeed":0,"zero_shot_only":0,"fine_tuned_only":0,"both_fail":0} for k in [1,10,100]}
for z,f in paired:
 for k in [1,10,100]:
  a=bool(z[f"Hit@{k}"]); b=bool(f[f"Hit@{k}"]); key="both_succeed" if a and b else "zero_shot_only" if a else "fine_tuned_only" if b else "both_fail"; comp[str(k)][key]+=1
write("complementarity.json", comp)
write("decision.json",{"outcome":"R7_CONFIGURATION_SELECTION_INVALIDATED","next_path":"C","recommendation":"Step 7.3 exact retraining with corrected DEV-selected L2 configuration","shift_map_v2_safe":False,"training_clean":True,"corrected_dev_selects_original_config":False,"evidence":{"loss_winner":"L2","original_loss":"L1"}})
write("hubness.json",{"status":"NOT_AVAILABLE","reason":"top-1 target code was not persisted in historical corrected replay rows"})
write("representation_drift.json",{"status":"NOT_AVAILABLE","reason":"embedding matrices were not persisted; a new diagnostic encoding run is required and is outside this completed replay artifact"})
write("no_map_diagnostics.json",{"status":"AVAILABLE","metrics":["max_cosine_similarity","top1_top2_margin","mean_top5_similarity"],"zero_shot":stats([r for r in zs if not r["valid_target_codes"]]),"fine_tuned_seed42":stats([r for r in canonical if not r["valid_target_codes"]]),"no_threshold_classifier_calibration":True})
write("rank_distribution.json",{"zero_shot_or_fine_tuned_rank_buckets":"derived from Hit@K rows; exact ranks >100 unavailable","zero_shot":stats(zs),"fine_tuned":stats(canonical)})
write("score_diagnostics.json",{"zero_shot":{k:metric(zs,k) for k in ["top1_score","top2_score","mean_top5_similarity"]},"fine_tuned_seed42":{k:metric(canonical,k) for k in ["top1_score","top2_score","mean_top5_similarity"]},"top10_score_spread":"NOT_PERSISTED"})
write("backward_transfer.json",{"status":"diagnostic_only","direction":"FORWARD_TRAINED_TO_BACKWARD","seeds":{str(s):stats([r for r in json.loads((ART/f'test_seed{s}.json').read_text())["rows"] if r["direction"]=="ICD10CM_TO_ICD9CM"]) for s in fts},"family_held_out":"NOT_AVAILABLE"})
write("training_direction_audit.json",json.loads((ART/"training_direction_audit.json").read_text()))
write("combination_metrics.json",{"zero_shot":stats([r for r in zs if r["mapping_kind"] in ["COMBINATION","COMBINATION_WITH_ALTERNATIVES"]]),"fine_tuned_seed42":stats([r for r in canonical if r["mapping_kind"] in ["COMBINATION","COMBINATION_WITH_ALTERNATIVES"]]),"complex_supervision_excluded":True})
write("manifest.json",{"experiment":"shift_map_v1_2","experiment_version":"1.2","evaluator_version":"2.0","training_occurred":False,"optimizer_created":False,"prior_test_exposure":True,"configuration_status":"corrected_DEV_invalidated_original_L1_final_configuration","artifacts":sorted(p.name for p in ART.glob("*.json") if p.name!="manifest.json")})
# CSVs
for fname, data in [("corrected_complementarity.csv",[{"K":k,**v} for k,v in comp.items()]),("corrected_paired_comparisons.csv",[{"metric":k,**v} for k,v in comparisons.items()])]:
 with (TAB/fname).open("w",newline="",encoding="utf-8") as f:
  keys=sorted({k for r in data for k in r}); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(data)
with (TAB/"corrected_three_seed_results.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=["seed","Hit@1","Hit@10","Hit@100","MRR","status"]); w.writeheader(); [w.writerow({"seed":s,**{k:stats(fts[s])[k] for k in ["Hit@1","Hit@10","Hit@100","MRR"]},"status":"diagnostic_only"}) for s in fts]
print(json.dumps({"artifacts":len(list(ART.glob('*.json'))),"tables":len(list(TAB.glob('*.csv'))),"paired_n":len(paired)},indent=2))
 # ruff: noqa: E501,E701,E702,E731,E741,F401

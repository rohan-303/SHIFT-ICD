# ruff: noqa
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'artifacts/experiments/shift_map_v1_4'; LED=EXP/'ledgers'; TAB=ROOT/'reports/tables/shift_map_v1_4'
KS=[1,5,10,25,50,100]

def rows(model): return json.loads((LED/f'{model}_forward_test.json').read_text())['rows']
def main():
    combo=[]
    for model in ['zero_shot','l1_seed42','l2_seed17']:
        rr=[r for r in rows(model) if r['mapping_kind'] in {'COMBINATION','COMBINATION_WITH_ALTERNATIVES'}]
        for k in KS:
            combo.append({'model':model,'k':k,'n_total':len(rr),'choice_list_recall':sum(bool(r.get(f'ChoiceListRecall@{k}')) for r in rr)/len(rr),'complete_scenario_retrieval':sum(bool(r.get(f'CompleteScenarioRetrieval@{k}')) for r in rr)/len(rr)})
    (EXP/'combination_transfer_final.json').write_text(json.dumps({'rows':combo,'combination_counts':{'COMBINATION':64,'COMBINATION_WITH_ALTERNATIVES':69}},indent=2)+'\n')
    with (TAB/'combination_transfer.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(combo[0])); w.writeheader(); w.writerows(combo)
    gate={'gate':'GO-A','step8_authorized':True,'candidate_generator':'frozen L2 seed17 epoch3','candidate_k':100,'basis':'DEV-selected checkpoint; diagnostics are characterization and TEST exposure is disclosed','caveat':'No clinical claims; L1 post-hoc TEST comparison does not change selection'}
    (EXP/'v2_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('combination_rows',len(combo),'gate',gate['gate'])
if __name__=='__main__': main()

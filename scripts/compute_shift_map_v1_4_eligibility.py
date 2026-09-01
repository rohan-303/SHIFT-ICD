# ruff: noqa
from __future__ import annotations
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; LED=ROOT/'artifacts/experiments/shift_map_v1_4/ledgers'; TAB=ROOT/'reports/tables/shift_map_v1_4'
def main():
 rows=[]
 for split in ['train','dev','test']:
  data=json.loads((LED/f'l2_seed17_forward_{split}.json').read_text())['rows']
  ordinary=[r for r in data if r['mapping_kind'] not in {'COMBINATION','COMBINATION_WITH_ALTERNATIVES','MULTI_SCENARIO','NO_MAP'}]
  eligible=sum(bool(set(r.get('valid_target_codes',[]))&set(r.get('ranked_codes',[])[:100])) for r in ordinary)
  rows.append({'split':split,'ordinary_sources':len(ordinary),'eligible_sources':eligible,'no_valid_candidate_sources':len(ordinary)-eligible,'combination_sources':sum(r['mapping_kind'] in {'COMBINATION','COMBINATION_WITH_ALTERNATIVES'} for r in data),'no_map_sources':sum(r['mapping_kind']=='NO_MAP' for r in data)})
 with (TAB/'v2_training_eligibility.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 (ROOT/'artifacts/experiments/shift_map_v1_4/v2_training_eligibility.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
 print(rows)
if __name__=='__main__': main()

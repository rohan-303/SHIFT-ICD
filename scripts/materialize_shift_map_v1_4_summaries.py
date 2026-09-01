# ruff: noqa
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'artifacts/experiments/shift_map_v1_4'; TAB=ROOT/'reports/tables/shift_map_v1_4'
def put(name,rows):
 rows=rows or [{'status':'NOT_COMPUTED'}]; keys=sorted({k for r in rows for k in r})
 with (TAB/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def main():
 d=json.loads((EXP/'representation_drift.json').read_text())
 target_rows=[{'group':k,**v} for k,v in d['train_positive_frequency'].items()]
 source_rows=[{'population':'all_targets',**d['overall']}]
 source_rows.extend({'population':k,**v} for k,v in d['by_training_exposure'].items())
 put('target_drift_by_training_exposure.csv',target_rows)
 put('source_vs_target_drift.csv',source_rows)
 manifest=json.loads((ROOT/'artifacts/candidates/shift_map_v2/manifest.json').read_text()); put('candidate_generation_manifest_summary.csv',manifest['rows'])
 print('summary_csvs_written')
if __name__=='__main__': main()

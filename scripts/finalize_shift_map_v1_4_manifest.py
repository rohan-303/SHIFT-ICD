# ruff: noqa
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'artifacts/experiments/shift_map_v1_4'; CAND=ROOT/'artifacts/candidates/shift_map_v2'
def digest(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 files=[]
 for base in [EXP,ROOT/'reports/tables/shift_map_v1_4',ROOT/'reports/figures/shift_map_v1_4',ROOT/'docs/experiments',ROOT/'docs/interfaces']:
  if base.exists():
   for p in base.rglob('*'):
    if p.is_file() and 'shift_map' in str(p).lower(): files.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size})
 manifest={'experiment_version':'1.4B','training_occurred':False,'evaluator_version':'2.0','benchmark_version':'1.0','canonical_schema_version':'1.0','checkpoint_hash':'7a859ff478d801f98a17fc966cb0ae81ba60362725add2735d91c7e01f4fe1d','target_corpus_hash':'a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a','candidate_manifest_sha256':digest(CAND/'manifest.json'),'files':files}
 (EXP/'final_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('manifest_files',len(files),'candidate_manifest_sha256',manifest['candidate_manifest_sha256'])
if __name__=='__main__': main()

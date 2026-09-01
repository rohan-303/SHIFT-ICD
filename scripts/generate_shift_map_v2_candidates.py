# ruff: noqa
from __future__ import annotations
import gzip, hashlib, json, time
from pathlib import Path
import pandas as pd
from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus

ROOT=Path(__file__).resolve().parents[1]; LED=ROOT/'artifacts/experiments/shift_map_v1_4/ledgers'; OUT=ROOT/'artifacts/candidates/shift_map_v2'; BENCH=ROOT/'data/benchmarks/cms_track_a/v1.0/all_examples.jsonl'
HASH='LOCAL_CHECKPOINT_HASH_PENDING'

def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 frame=pd.read_parquet(ROOT/'data/processed/cms/2018_gem/normalized_rows.parquet')
 target_map=build_target_corpus(frame, FORWARD).as_dict()
 target_hash=build_target_corpus(frame, FORWARD).corpus_hash
 checkpoint_files=sorted((ROOT/'artifacts/models/shift_map_v1/v1_3_final_l2_n1_seed17/epoch_3').rglob('*'))
 h=hashlib.sha256()
 for file in checkpoint_files:
     if file.is_file(): h.update(str(file.relative_to(ROOT)).encode()); h.update(file.read_bytes())
 global HASH
 HASH=h.hexdigest()
 examples={}
 for line in BENCH.open(encoding='utf8'):
  if line.strip():
   x=json.loads(line); examples[x['benchmark_id']]=x
 produced=[]
 for split in ('train','dev','test'):
  src=json.loads((LED/f'l2_seed17_forward_{split}.json').read_text())['rows']; path=OUT/f'forward_stratified_{split}_k100.jsonl.gz'; nrows=0
  with gzip.open(path,'wt',encoding='utf8') as f:
   for r in src:
    x=examples[r['benchmark_id']]; codes=r.get('ranked_codes',[])[:100]; scores=r.get('ranked_scores',[])[:100]
    for rank,(code,score) in enumerate(zip(codes,scores,strict=False),1):
     f.write(json.dumps({'benchmark_id':r['benchmark_id'],'direction':r['direction'],'source_code':x['source_code'],'source_description':x['source_label'],'target_code':code,'target_description':target_map.get(code,''),'retriever_rank':rank,'retriever_score':score,'candidate_k':100,'mapping_kind':r['mapping_kind'],'split':split,'source_family':x.get('source_family'),'target_family':None,'candidate_is_gold':code in set(r.get('valid_target_codes',[]))},separators=(',',':'))+'\n'); nrows+=1
  produced.append({'split':split,'source_count':len(src),'candidate_k':100,'candidate_row_count':nrows,'file':path.name,'sha256':sha(path),'retriever_checkpoint_hash':HASH,'target_corpus_hash':target_hash,'evaluator_version':'2.0','generated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 (OUT/'manifest.json').write_text(json.dumps({'schema_version':'1.0','candidate_generator':'SHIFT-MAP v1.3 corrected L2 seed17 epoch3','rows':produced},indent=2)+'\n')
 (ROOT/'artifacts/experiments/shift_map_v1_4/candidate_generation_manifest_summary.json').write_text(json.dumps({'rows':produced},indent=2)+'\n')
 print(json.dumps(produced,indent=2))
if __name__=='__main__': main()

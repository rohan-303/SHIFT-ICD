# ruff: noqa
from __future__ import annotations
import argparse, gzip, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from shift_icd.reranking.inference import ChunkManifest, ThermalGuard, ThermalState
from shift_icd.reranking.shift_map_v2 import ndcg_at_k

ROOT=Path(__file__).resolve().parents[1]
CAND=ROOT/'artifacts/candidates/shift_map_v2/forward_stratified_dev_k100.jsonl.gz'
OUT=ROOT/'artifacts/experiments/shift_map_v2_compute/zero_shot_dev_chunks'
MODEL_REVISION='71caf65d4927987813984f54c284405a13fcca49'
ORDINARY={'SINGLE_EXACT','SINGLE_APPROXIMATE','ALTERNATIVE'}

def load_groups():
 groups={}
 with gzip.open(CAND,'rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line); groups.setdefault(r['benchmark_id'],{'benchmark_id':r['benchmark_id'],'rows':[]})['rows'].append(r)
 out=list(groups.values())
 for g in out:g['rows'].sort(key=lambda r:r['retriever_rank'])
 return out

def temp():
 import subprocess
 return float(subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().splitlines()[0])

def power_plugged():
 import psutil
 battery=psutil.sensors_battery()
 return bool(battery and battery.power_plugged)

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--chunk-sources',type=int,default=50);p.add_argument('--max-length',type=int,default=96);a=p.parse_args()
 groups=load_groups(); candidate_hash=hashlib.sha256(CAND.read_bytes()).hexdigest(); OUT.mkdir(parents=True,exist_ok=True)
 mpath=OUT/'manifest.json'; m=ChunkManifest.load(mpath) if mpath.exists() else ChunkManifest(mpath,len(groups)*100)
 guard=ThermalGuard(temp,soft_pause=80,hard_stop=83,resume=72); tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,revision=MODEL_REVISION); model=AutoModelForSequenceClassification.from_pretrained(a.snapshot,local_files_only=True,revision=MODEL_REVISION).to('cuda:0').eval()
 start=time.perf_counter(); active=0.0; interruptions=0
 for gs in range(0,len(groups),a.chunk_sources):
  ge=min(gs+a.chunk_sources,len(groups)); cid=f'sources-{gs:06d}-{ge:06d}'
  if cid in m.chunks: continue
  if guard.check()==ThermalState.HARD_STOP: break
  chunk_start=time.perf_counter(); rows=[]; pairs=[]
  for g in groups[gs:ge]: pairs.extend((r['source_description'],r['target_description']) for r in g['rows'])
  vals=[]
  for i in range(0,len(pairs),a.batch_size):
   if not power_plugged():
    m.save(); print('POWER_LOSS_STOP',flush=True); break
   state=guard.check()
   while state in (ThermalState.COOLDOWN, ThermalState.HARD_STOP):
    interruptions+=1; time.sleep(2); state=guard.check()
   enc=tok(pairs[i:i+a.batch_size],padding='longest',truncation=True,max_length=a.max_length,return_tensors='pt');enc={k:v.to('cuda:0') for k,v in enc.items()}
   with torch.inference_mode(): vals.extend(model(**enc).logits[:,0].float().cpu().tolist())
  if len(vals)!=len(pairs): m.save(); break
  pos=0
  for g in groups[gs:ge]:
   n=len(g['rows']); scores=np.asarray(vals[pos:pos+n]);pos+=n; order=np.argsort(-scores,kind='stable'); rr=[g['rows'][int(j)] for j in order]; codes=[r['target_code'] for r in rr]; gold={r['target_code'] for r in g['rows'] if r['candidate_is_gold']}; first=next((i+1 for i,c in enumerate(codes) if c in gold),None); rows.append({'benchmark_id':g['benchmark_id'],'mapping_kind':g['rows'][0]['mapping_kind'],'ranked_codes':codes,'scores':[float(scores[int(j)]) for j in order],'gold_codes':sorted(gold),'mrr':1/first if first else 0.0,'hits':{str(k):float(bool(set(codes[:k])&gold)) for k in (1,3,5,10,25,50,100)},'ndcg10':ndcg_at_k(gold,codes,10)})
  payload=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode(); (OUT/f'chunk_{cid}.jsonl').write_bytes(payload); m.add_chunk(cid,gs*100,ge*100,payload);m.save();active+=time.perf_counter()-chunk_start
  print(f'completed_sources={ge}/{len(groups)} temperature={guard.max_temperature}',flush=True)
 allrows=[]
 for path in sorted(OUT.glob('chunk_sources-*.jsonl')):
  allrows.extend(json.loads(x) for x in path.read_text(encoding='utf8').splitlines())
 ordinary=[r for r in allrows if r['mapping_kind'] in ORDINARY]; metrics={'n_all_sources':len(allrows),'n_ordinary_sources':len(ordinary)}
 for k in (1,3,5,10,25,50,100):metrics[f'Hit@{k}']=float(np.mean([r['hits'][str(k)] for r in ordinary])) if ordinary else 0.0
 metrics['MRR']=float(np.mean([r['mrr'] for r in ordinary])) if ordinary else 0.0;metrics['NDCG@10']=float(np.mean([r['ndcg10'] for r in ordinary])) if ordinary else 0.0
 result={'status':'COMPLETE' if len(allrows)==len(groups) else ('THERMAL_HARD_STOP' if guard.hard_stop_triggered else 'INCOMPLETE'),'metrics':metrics,'expected_sources':len(groups),'completed_sources':len(allrows),'expected_pairs':len(groups)*100,'completed_pairs':m.completed_pairs,'candidate_file_sha256':candidate_hash,'batch_size':a.batch_size,'chunk_sources':a.chunk_sources,'max_length':a.max_length,'active_seconds':active,'wall_seconds':time.perf_counter()-start,'cooldown_seconds':guard.cooldown_seconds,'pause_count':guard.pause_count,'maximum_temperature':guard.max_temperature,'interruptions':interruptions,'hit100_candidate_invariant':'REQUIRES_COVERAGE_COMPARISON'}
 (ROOT/'artifacts/experiments/shift_map_v2_compute/zero_shot_dev_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

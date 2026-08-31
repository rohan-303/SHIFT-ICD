# ruff: noqa: E501,E702,E703,I001
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts/experiments/shift_map_v1_2'; F=ROOT/'reports/figures/shift_map_v1_2'; F.mkdir(parents=True,exist_ok=True)
z=json.loads((A/'zero_shot_test.json').read_text())['populations']['ICD9CM_TO_ICD10CM']['rows']; f=json.loads((A/'test_seed42.json').read_text())['rows']; f=[r for r in f if r['direction']=='ICD9CM_TO_ICD10CM']
def m(rows,k): return sum(r[k] for r in rows if r[k] is not None)/sum(r[k] is not None for r in rows)
def save(name,title,x,y,labels):
 fig,ax=plt.subplots(figsize=(7,4)); ax.bar(x,y); ax.set_title(title+' (diagnostic only)'); ax.set_xticks(range(len(x)),labels); ax.set_ylim(0,1); fig.tight_layout(); fig.savefig(F/name,dpi=150); plt.close(fig)
ks=['Hit@1','Hit@10','Hit@100']; save('zero_shot_vs_shift_map_hit_at_k.png','Zero-shot vs corrected SHIFT-MAP',ks,[m(z,k) for k in ks],[m(f,k) for k in ks]);
# canonical versus seeds
seeds={}
for s in [17,42,2026]:
 rows=[r for r in json.loads((A/f'test_seed{s}.json').read_text())['rows'] if r['direction']=='ICD9CM_TO_ICD10CM']; seeds[str(s)]=m(rows,'Hit@10')
save('corrected_three_seed_comparison.png','Corrected three-seed Hit@10',list(seeds),list(seeds.values()),list(seeds))
# remaining requested names are explicit diagnostic placeholders because slice plot inputs are not persisted separately
for name,title in [('lexical_difficulty.png','Lexical difficulty'),('mapping_kind.png','Mapping kind'),('combination_retrieval.png','Combination retrieval'),('family_held_out.png','Family-held-out'),('backward_transfer.png','Backward transfer'),('success_regression_complementarity.png','Success/regression complementarity'),('valid_target_rank_distribution.png','Valid-target rank distribution'),('representation_drift.png','Representation drift')]:
 fig,ax=plt.subplots(figsize=(7,4)); ax.text(.5,.5,'See corrected Step 7.2 report/artifacts\nAnalysis unavailable or diagnostic-only',ha='center',va='center'); ax.set_axis_off(); ax.set_title(title+' — status disclosure'); fig.savefig(F/name,dpi=150); plt.close(fig)
print('figures',len(list(F.glob('*.png'))))
 # ruff: noqa: E501,E702,E703,I001

"""Independent cohort/hash checks and scalar metric recomputation before reporting."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from gate import validate,ARMS

def main():
 started=time.time();audit={};pre=json.loads((O/'preflight.json').read_text())
 for ds in ['mpd','m4a']:
  for p,h in pre['datasets'][ds]['source_hashes'].items():assert sha(p)==h,'v1/source changed: '+p
  co=json.loads((O/ds/'cohort.json').read_text());assert len(co['pilot'])==100 and len(co['gate_rows'])==200
  assert len({r['source_history_index'] for r in co['gate_rows']})==200
  for r in co['gate_rows']:assert len(r['history'])==15 and len(r['targets'])==5 and r['condition_item']==r['targets'][0] and r['seed']==42000+r['source_history_index']
  x=paths(ds);ids=json.loads((x['catalog']/'catalog_item_ids.json').read_text());lookup={str(i):j for j,i in enumerate(ids)};emb=np.load(x['catalog']/'catalog_mert_embeddings_l2.npy').astype(float);emb/=np.linalg.norm(emb,axis=1,keepdims=True);target=emb[[[lookup[i] for i in r['targets']] for r in co['gate_rows']]];history=emb[[[lookup[i] for i in r['history']] for r in co['gate_rows']]];per=np.load(O/ds/'gate-per-context.npz');report=json.loads((O/ds/'gate-report.json').read_text())
  arm_audit={}
  for arm in ARMS:
   validate(ds,arm);b=O/ds/'gate'/arm;pred=np.stack([np.load(b/'metrics/mert-rows'/f'{i:04d}.npz')['embedding'] for i in range(200)]).astype(float);pred/=np.linalg.norm(pred,axis=1,keepdims=True);scores=[];hist=[]
   for p,t,h in zip(pred,target,history):
    cost=1-p.reshape(1,-1)@t.T;r,c=linear_sum_assignment(cost);scores.append(1-cost[r,c].sum());hist.append(float((h@p).mean()))
   np.testing.assert_allclose(scores,per[arm+'__mert_oas'],atol=1e-7);np.testing.assert_allclose(hist,per[arm+'__history_fit'],atol=1e-7)
   for k in ['mert_oas','history_fit','content_fit_v1','content_fit_v2']:
    assert abs(per[arm+'__'+k].mean()-report['arms'][arm][k]['mean'])<1e-10
   arm_audit[arm]=dict(clips=200,mert_oas_independent=True,history_fit_independent=True,audio_hashes_verified=True)
  for other in ['none','v1','mmr']:
   for k in ['mert_oas','history_fit','content_fit_v1','content_fit_v2']:
    assert abs((per['v2__'+k]-per[other+'__'+k]).mean()-report['paired']['v2 minus '+other][k]['mean'])<1e-10
  audit[ds]=arm_audit
 hashes={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.html'] and not any(x in p.parts for x in ['logs','running','done','failed','locks','processes','preflight-before-lyrics-path-fix']) and p.name!='verification.json'}
 atomic(O/'verification.json',dict(passed=True,datasets=audit,total_clips=1600,stage_B_started=False,personalization_v2='deferred by author',elapsed_seconds=time.time()-started,hashes=hashes))

if __name__=='__main__':main()

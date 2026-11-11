import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--setting',choices=['full_catalog','sampled100'],default='full_catalog');a=p.parse_args()
for ds in ['mpd','m4a']:
 cpath=R/'outputs/sampled-candidates-20261110'/ds/'candidates.json';c=json.loads(cpath.read_text());idx=sample_indices(ds);x=paths(ds);itemc=json.loads((x['cues']/'item2cues.json').read_text());vocab=json.loads((x['cues']/'cue_vocab.json').read_text())
 ids=c['catalog_ids'];codes=np.load(x['prepared']/'vectors/catalog_semantic_tokens.npy');assert len(ids)==len(codes);code_map=dict(zip(ids,codes.tolist()))
 sources=source_plans(ds,a.setting);plans={k:json.loads(v.read_text()) for k,v in sources.items()}
 for k,q in plans.items():assert len(q['predictions']['item_ids'])==941 and q['predictions']['target_item_ids']==c['truth']
 cohort={'schema':'genplaylist-list5-v1','dataset':ds,'candidate_setting':a.setting,'lists':100,'items_per_list':5,'renderings_per_arm':500,'sampling_seed':20261111,'indices':idx,'source_test_indices':[c['indices'][i] for i in idx],'sampling_population':941,'source_cohort_sha256':sha(cpath),'rows':[dict(flat_index=k*5+j,list_index=k,source_history_index=i,source_test_index=c['indices'][i],item_position=j,history=c['histories'][i],target=c['truth'][i][j],seed=seed(i,j)) for k,i in enumerate(idx) for j in range(5)]}
 immutable(O/ds/'cohort.json',cohort)
 for arm in ARMS:
  srcarm='ddbc_joint' if arm in ['drop','retrieved'] else arm
  rows=[]
  for row in cohort['rows']:
   i=row['source_history_index'];j=row['item_position'];r=dict(row);r.update(arm=arm,system=SYSTEMS[arm]);pr=plans.get(srcarm,{}).get('predictions')
   if pr:
    song=pr['item_ids'][i][j];cues=pr['cue_ids'][i][j] if arm in ['ddbc_joint','tiger_joint'] else [] if arm=='drop' else itemc[song][:8]
    r.update(predicted_item_id=song,semantic_token_ids=code_map[song],cue_ids=cues)
   else:r.update(predicted_item_id=None,semantic_token_ids=None,cue_ids=history_random(r['history'],itemc,vocab,ds,i,j) if arm=='history_random' else [])
   if arm!='drop' and pr:assert len(r['cue_ids'])==8
   rows.append(r)
  immutable(O/ds/arm/'render-plan.json',dict(schema='genplaylist-list5-render-plan-v1',cohort_sha256=sha(O/ds/'cohort.json'),setting=a.setting,arm=arm,source_plan=None if srcarm not in sources else str(sources[srcarm]),source_plan_sha256=None if srcarm not in sources else sha(sources[srcarm]),rows=rows))
 print(ds,len(idx),'lists',len(rows),'items',a.setting,flush=True)

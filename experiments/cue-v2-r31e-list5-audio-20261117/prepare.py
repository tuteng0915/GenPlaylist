import argparse,json
from common import *

p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);a=p.parse_args()
ds=a.dataset
cpath=R/'outputs/sampled-candidates-20261110'/ds/'candidates.json'
c=json.loads(cpath.read_text());idx=sample_indices(ds);x=paths(ds);itemc,vocab=v2_cues(ds)
ids=c['catalog_ids'];codes=np.load(x['prepared']/'vectors/catalog_semantic_tokens.npy');assert len(ids)==len(codes);code_map=dict(zip(ids,codes.tolist()))
sources=source_plans(ds);plans={k:json.loads(v.read_text()) for k,v in sources.items()}
for k,q in plans.items():
 pr=q['predictions'];assert len(pr['item_ids'])==941 and pr['target_item_ids']==c['truth'],k
 assert all(len(row)==5 and len(set(row))==5 for row in pr['item_ids']),k
cohort={'schema':'cue-v2-r31e-list5-v1','dataset':ds,'candidate_setting':'full_catalog','lists':100,'items_per_list':5,'renderings_per_arm':500,'sampling_seed':20261111,'indices':idx,'source_test_indices':[c['indices'][i] for i in idx],'sampling_population':941,'source_cohort_sha256':sha(cpath),'rows':[dict(flat_index=k*5+j,list_index=k,source_history_index=i,source_test_index=c['indices'][i],item_position=j,history=c['histories'][i],target=c['truth'][i][j],seed=seed(i,j)) for k,i in enumerate(idx) for j in range(5)]}
immutable(O/ds/'cohort.json',cohort)
arms=[x for x in ARMS if x!='tiger_sid' or ds=='m4a']
for arm in arms:
 src='ddbc_joint' if arm in ['ddbc_catalog','ddbc_nocue'] else arm
 pr=plans[src]['predictions'];rows=[]
 for row in cohort['rows']:
  i=row['source_history_index'];j=row['item_position'];song=str(pr['item_ids'][i][j]);r=dict(row)
  if arm=='ddbc_joint': cues=pr['cue_ids'][i][j]
  elif arm=='ddbc_nocue': cues=[]
  else: cues=itemc[song][:8]
  semantic=pr.get('semantic_token_ids',[[None]*5 for _ in range(941)])[i][j] or code_map[song]
  assert arm=='ddbc_nocue' or len(cues)==8,(arm,ds,i,j,len(cues))
  r.update(arm=arm,system=SYSTEMS[arm],predicted_item_id=song,semantic_token_ids=semantic,cue_ids=cues)
  rows.append(r)
 plan=dict(schema='cue-v2-r31e-list5-render-plan-v1',cohort_sha256=sha(O/ds/'cohort.json'),arm=arm,source_plan=str(sources[src]),source_plan_sha256=sha(sources[src]),cue_vocab_sha256=sha(R/'outputs/cue-v2-r31e-20261113/cue_vocab.json'),item2cues_sha256=sha(R/'outputs/cue-v2-r31e-20261113'/ds/'item2cues.json'),rows=rows)
 immutable(O/ds/arm/'render-plan.json',plan)
print(json.dumps({'dataset':ds,'arms':arms,'lists':100,'clips_per_arm':500,'sources':{k:str(v) for k,v in sources.items()}},indent=2))

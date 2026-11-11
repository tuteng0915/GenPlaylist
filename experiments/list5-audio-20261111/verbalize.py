import argparse,random
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
x=paths(a.dataset);base=O/a.dataset/a.arm;plan=json.loads((base/'render-plan.json').read_text());rows=plan['rows'];assert len(rows)==500 and 0<=a.start<a.start+a.count<=500
v=module('list5_shared_verbal',CODE/'scripts/run_end_to_end_verbalization.py');verbal=v._load_verbalization_module();root=base/'verbalization';system=SYSTEMS[a.arm]
lookup={str(i):int(j) for i,j in json.loads((x['art']/'item_id_to_row.json').read_text()).items()};emb=np.load(x['art']/'catalog_item_embeddings.npy').astype(np.float32);weights=np.load(x['art']/'rvq_codebook_weights.npy')
meta=v.CatalogItem.load_catalog(str(x['data']/'catalog_metadata.json'));meta={m.item_id:m for m in meta} if isinstance(meta,list) else meta;ordered=[None]*len(lookup)
for i,j in lookup.items():ordered[j]=meta[i]
itemc=json.loads((x['cues']/'item2cues.json').read_text());vocab=json.loads((x['cues']/'cue_vocab.json').read_text())
assets={str(f):sha(f) for f in [x['data']/'catalog_metadata.json',x['art']/'catalog_item_embeddings.npy',x['art']/'rvq_codebook_weights.npy',x['cues']/'item2cues.json',x['cues']/'cue_vocab.json',CODE/'scripts/run_end_to_end_verbalization.py']}
immutable(root/'verbalization_manifest.json',dict(schema='genplaylist-list5-verbalization-v2',model_name='Qwen/Qwen3-4B-Instruct-2507',model_revision=REV,seed=42,decoding='greedy',systems=[system],examples=500,lists=100,positions=5,reference_items=15,render_plan_sha256=sha(base/'render-plan.json'),assets=assets))
random.seed(42);np.random.seed(42)
import torch
torch.manual_seed(42);caller=None;reuse=old_root(a.dataset,a.arm)
for row in rows[a.start:a.start+a.count]:
 k=row['flat_index'];dest=root/system/f'{k:04d}.json'
 if dest.exists():
  prev=json.loads(dest.read_text());assert prev['render_row']==row;continue
 prior=None;oi=row['source_history_index'];slot=row['item_position']
 if reuse and (slot==0 or a.arm in ['direct','profile']):
  old=reuse/'verbalization'/system/f'{oi:04d}.json';manifest=reuse/'verbalization/verbalization_manifest.json'
  if old.exists() and manifest.exists():
   q=json.loads(old.read_text());m=json.loads(manifest.read_text())
   inputs=m.get('inputs',{});expected_inputs={'catalog_embeddings_sha256':sha(x['art']/'catalog_item_embeddings.npy'),'catalog_metadata_sha256':sha(x['data']/'catalog_metadata.json'),'cue_vocab_sha256':sha(x['cues']/'cue_vocab.json'),'item2cues_sha256':sha(x['cues']/'item2cues.json')}
   compatible=m.get('model_revision')==REV and m.get('model_name')=='Qwen/Qwen3-4B-Instruct-2507' and m.get('decoding')=='greedy' and m.get('seed')==42 and m.get('reference_items')==15 and all(inputs.get(k)==v for k,v in expected_inputs.items())
   if compatible and q['reference_item_ids']==row['history'] and q.get('retrieved_item_id')==row['predicted_item_id'] and q['cue_ids']==row['cue_ids']:prior=(old,q)
 if prior:record=dict(prior[1]);record.update(reused_verbalization=str(prior[0]),reused_verbalization_sha256=sha(prior[0]))
 else:
  refs=row['history'];mu,sigma=v._reference_structure(refs,lookup,emb)
  tiny=None
  if row['predicted_item_id'] is not None:tiny={'predictions':{'item_ids':[[row['predicted_item_id']]],'semantic_token_ids':[[row['semantic_token_ids']]],'cue_ids':[[row['cue_ids'] if a.arm!='drop' else itemc[row['predicted_item_id']][:8]]]}}
  g=v._generated_item(system=system,example_index=0,reference_ids=refs,mu_c=mu,sigma_c2=sigma,plan=tiny,codebook_weights=weights,item2cues=itemc,cue_intervention='drop' if a.arm=='drop' else 'none')
  if a.arm=='history_random':g.cue_ids=row['cue_ids'];g.validate()
  if caller is None:caller=v.LocalQwen(model_name='Qwen/Qwen3-4B-Instruct-2507',revision=REV,device='cuda');assert caller.model_revision==REV
  profile_extra={}
  if a.arm=='profile':
   pc=module('list5_profile_common',R/'experiments/pipeline-baselines-20261110/common.py');anchors=verbal.knn_verbalize(mu,emb,ordered,5);prompt=pc.profile_prompt([meta[t].to_prompt_line() for t in refs],[t.to_prompt_line() for t in anchors]);raw=caller(prompt,system=pc.PROFILE_SYSTEM);profile=pc.parse_profile(raw);g.context_prefix.raw_input='Create original music consistent with this history-derived musical preference profile: '+json.dumps(profile,ensure_ascii=False,sort_keys=True);profile_extra=dict(preference_profile=profile,profile_prompt=prompt,profile_raw_response=raw)
  result=verbal.verbalize(generated=g,catalog_embs=emb,catalog_metadata=ordered,k=5,cue_vocab=vocab,llm_call=caller)
  record=dict(reference_item_ids=refs,retrieved_item_id=row['predicted_item_id'],cue_ids=g.cue_ids,cue_terms=result['cue_terms'],neighbor_item_ids=[t.item_id for t in result['neighbors']],style_item_ids=[t.item_id for t in result['style_summary']],music_attributes=result['music_attributes'],lyric_draft=result['lyric_draft'],**profile_extra)
 record.update(result_schema='genplaylist-list5-verbalization-record-v2',system=system,example_index=k,target_item_id=row['target'],render_row=row,seed=row['seed'])
 assert record['cue_ids']==row['cue_ids'];atomic(dest,record);print('VERBAL',a.dataset,a.arm,k,flush=True)

import argparse,random
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
x=paths(a.dataset);base=O/a.dataset/a.arm;plan=json.loads((base/'render-plan.json').read_text());rows=plan['rows'];assert len(rows)==500 and 0<=a.start<a.start+a.count<=500
v=module('list5_shared_verbal',CODE/'scripts/run_end_to_end_verbalization.py');verbal=v._load_verbalization_module();root=base/'verbalization';system=SYSTEMS[a.arm]
lookup={str(i):int(j) for i,j in json.loads((x['art']/'item_id_to_row.json').read_text()).items()};emb=np.load(x['art']/'catalog_item_embeddings.npy').astype(np.float32);weights=np.load(x['art']/'rvq_codebook_weights.npy')
meta=v.CatalogItem.load_catalog(str(x['data']/'catalog_metadata.json'));meta={m.item_id:m for m in meta} if isinstance(meta,list) else meta;ordered=[None]*len(lookup)
for i,j in lookup.items():ordered[j]=meta[i]
itemc,vocab=v2_cues(a.dataset)
v2root=R/'outputs/cue-v2-r31e-20261113'
assets={str(f):sha(f) for f in [x['data']/'catalog_metadata.json',x['art']/'catalog_item_embeddings.npy',x['art']/'rvq_codebook_weights.npy',v2root/a.dataset/'item2cues.json',v2root/'cue_vocab.json',CODE/'scripts/run_end_to_end_verbalization.py']}
immutable(root/'verbalization_manifest.json',dict(schema='genplaylist-list5-verbalization-v2',model_name='Qwen/Qwen3-4B-Instruct-2507',model_revision=REV,seed=42,decoding='greedy',systems=[system],examples=500,lists=100,positions=5,reference_items=15,render_plan_sha256=sha(base/'render-plan.json'),assets=assets))
random.seed(42);np.random.seed(42)
import torch
torch.manual_seed(42);caller=None
for row in rows[a.start:a.start+a.count]:
 k=row['flat_index'];dest=root/system/f'{k:04d}.json'
 if dest.exists():
  prev=json.loads(dest.read_text());assert prev['render_row']==row;continue
 if True:
  refs=row['history'];mu,sigma=v._reference_structure(refs,lookup,emb)
  tiny=None
  if row['predicted_item_id'] is not None:tiny={'predictions':{'item_ids':[[row['predicted_item_id']]],'semantic_token_ids':[[row['semantic_token_ids']]],'cue_ids':[[row['cue_ids'] if a.arm!='ddbc_nocue' else itemc[row['predicted_item_id']][:8]]]}}
  g=v._generated_item(system=system,example_index=0,reference_ids=refs,mu_c=mu,sigma_c2=sigma,plan=tiny,codebook_weights=weights,item2cues=itemc,cue_intervention='drop' if a.arm=='ddbc_nocue' else 'none')
  if caller is None:caller=v.LocalQwen(model_name='Qwen/Qwen3-4B-Instruct-2507',revision=REV,device='cuda');assert caller.model_revision==REV
  result=verbal.verbalize(generated=g,catalog_embs=emb,catalog_metadata=ordered,k=5,cue_vocab=vocab,llm_call=caller)
  record=dict(reference_item_ids=refs,retrieved_item_id=row['predicted_item_id'],cue_ids=g.cue_ids,cue_terms=result['cue_terms'],neighbor_item_ids=[t.item_id for t in result['neighbors']],style_item_ids=[t.item_id for t in result['style_summary']],music_attributes=result['music_attributes'],lyric_draft=result['lyric_draft'])
 record.update(result_schema='genplaylist-list5-verbalization-record-v2',system=system,example_index=k,target_item_id=row['target'],render_row=row,seed=row['seed'])
 assert record['cue_ids']==row['cue_ids'];atomic(dest,record);print('VERBAL',a.dataset,a.arm,k,flush=True)

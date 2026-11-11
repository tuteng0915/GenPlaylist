import argparse,ast,itertools
from collections import Counter
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cue','training','test','smoke'],required=True);a=p.parse_args()
tree=ast.parse((R/'experiments/two-dataset-fair-20261108/catalog_report.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
oas=module('preserve_oas',R/'experiments/list5-audio-20261111/oas.py')
allout={};selection={}
for ds in ['mpd','m4a']:
 cfg,tok,raw,split,sp=setup(ds,'joint');itemc=json.loads(Path(cfg.item2cues_path).read_text());ids=sorted(tok.item_id_to_row,key=tok.item_id_to_row.get)
 emb=np.load(cfg.catalog_embeddings_path);mapping=json.loads(Path(cfg.item_id_to_row_path).read_text())
 ns=dict(np=np,Counter=Counter,set_ids=set(ids));exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_score','exec'),ns)
 partition='smoke' if a.phase=='smoke' else 'test' if a.phase=='test' else 'validation';cases=[]
 for f in (O/'eval'/ds).glob(f'*/*/{partition}/*/complete.json'):
  identity=json.loads(f.read_text())['identity']
  if a.phase=='cue' and identity['arm']!='existing':continue
  cases.append(f)
 per={};identities={};out={'dataset':ds,'phase':a.phase,'metrics':{},'paired':{},'sources':{}}
 for f in sorted(cases):
  d=f.parent;identity=json.loads((d/'identity.json').read_text());plan=json.loads((d/'plan.json').read_text())['predictions'];idx=identity['indices']
  selected=raw['test' if partition=='test' else 'train'].select(idx);hist=[[str(s) for s in r[:15]] for r in selected['item_seq']];truth=[[str(s) for s in r[15:]] for r in selected['item_seq']]
  assert truth==plan['target_item_ids'];pred=plan['item_ids'];cue=plan['cue_ids'] if identity['arm']!='sid' else [[itemc[s][:8] for s in r] for r in pred]
  scores,vals=ns['score'](pred,truth,hist,ids,cue,itemc);scores.pop('cue_song_grounding_f1',None);vals.pop('cue_song_grounding_f1',None)
  rr=[sum((Counter(x)&Counter(y)).values())/5 for x,y in zip(pred,truth)]
  ff=[sum((Counter(v for row in x for v in row)&Counter(v for s in y for v in itemc[s][:8])).values())/40 for x,y in zip(cue,truth)]
  np.testing.assert_allclose(rr,vals['recall_at_5'],atol=1e-12);np.testing.assert_allclose(ff,vals['cue_f1'],atol=1e-12)
  features=emb[np.array([[mapping[s] for s in r] for r in pred])];target=emb[np.array([[mapping[s] for s in r] for r in truth])];match=oas.assignment_oas(features,target)
  perms=np.array(list(itertools.permutations(range(5))));brute=np.array([m[np.arange(5)[None,:],perms].mean(1).max() for m in match['pairwise_cosine']]);np.testing.assert_allclose(brute,match['similarity'],atol=1e-12)
  vals['clhe_oas']=brute;scores['clhe_oas']=float(brute.mean());scores['diversity']=scores.pop('catalog_coverage')
  key=f"{identity['arm']}-step{identity['step']}-alpha{identity['alpha']:g}";np.savez_compressed(d/'audited-per-context.npz',**vals);per[key]=vals;identities[key]=idx;out['metrics'][key]=scores;out['sources'][key]={'identity':identity,'plan_sha256':sha(d/'plan.json'),'cue_source_sha256':sha(cfg.item2cues_path),'embedding_source_sha256':sha(cfg.catalog_embeddings_path),'cue_semantics':'catalog cues of recommended songs' if identity['arm']=='sid' else 'predicted cues'}
  groups=[str(g).split(':joint5:')[0] for g in selected['bundle']] if ds=='mpd' else [__import__('re').search(r'm4a-train-([a-f0-9]+)-',str(g)).group(1) for g in selected['bundle']] if partition!='test' else list(map(str,idx))
  unique=list(dict.fromkeys(groups));assign=np.array([unique.index(g) for g in groups]);counts=np.bincount(assign)
 def ci(v):
  rng=np.random.default_rng(42);sums=np.bincount(assign,weights=v);boot=[]
  for _ in range(100):
   ix=rng.integers(len(unique),size=(100,len(unique)));boot.extend(sums[ix].sum(1)/counts[ix].sum(1))
  return np.quantile(boot,[.025,.975]).tolist()
 baseline='existing-step1000-alpha1' if a.phase in ('cue','smoke') else 'sid-step1000-alpha1'
 if baseline in per:
  for key,v in per.items():
   assert identities[key]==identities[baseline], 'Paired contexts differ'
   out['paired'][key+' minus '+baseline]={m:dict(delta=float((v[m]-per[baseline][m]).mean()),ci95=ci(v[m]-per[baseline][m])) for m in ['recall_at_5','cue_f1','clhe_oas']}
 for left,right in [('preserve-step1000-alpha1','plain-step1000-alpha1')]+[(k,'existing-step1000-alpha1') for k in per if k.startswith('existing-')]:
  if left in per and right in per:
   assert identities[left]==identities[right]
   out['paired'][left+' minus '+right]={m:dict(delta=float((per[left][m]-per[right][m]).mean()),ci95=ci(per[left][m]-per[right][m])) for m in ['recall_at_5','cue_f1','clhe_oas']}
 if a.phase=='cue':
  assert len(per)==4;ref=out['metrics'][baseline];eligible=[]
  for alpha in [0,.25,.5,1]:
   key=f'existing-step1000-alpha{alpha:g}';m=out['metrics'][key]
   if m['cue_f1']>=ref['cue_f1']-.005 and m['diversity']>=ref['diversity']*.9 and m['catalog_entropy']>=ref['catalog_entropy']-.1:eligible.append((m['recall_at_5'],alpha))
  chosen=max(eligible,key=lambda x:(x[0],x[1]))[1];selection[ds]=dict(alpha=chosen,reason='max validation Recall under predeclared point-estimate guardrails; ties prefer original alpha1; exploratory, not noninferiority proof')
 out['contexts']=len(idx);out['clusters']=len(unique);out['verification']='independent multiset Recall/Cue F1 and all120 OAS permutations';allout[ds]=out
save(O/f'{a.phase}-report.json',allout)
if a.phase=='cue':save(O/'alpha-selection.json',selection)
print('REPORT_COMPLETE',a.phase,selection,flush=True)

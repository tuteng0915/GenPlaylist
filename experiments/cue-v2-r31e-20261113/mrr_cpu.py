import collections,json,sys,time,os
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer
R=Path('/storage_fast/ttu/GenPlaylist');V=R/'outputs/cue-v2-r31e-20261113';S=R/'outputs/cue-v2-stageA-20261112';sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'));from common import normalized
from build import unit
torch.set_num_threads(3);model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device='cpu',local_files_only=True);vocab=json.loads((V/'cue_vocab.json').read_text())
for ds in ['mpd','m4a']:
 t=time.time();inp=json.loads((S/ds/'inputs.json').read_text());m=json.loads((V/ds/'item2cues.json').read_text());ids=sorted(inp);assert set(ids)==set(m)
 target=unit(model.encode([' '.join([inp[i]['genre'],' '.join(inp[i]['tags']),inp[i]['mood'],inp[i]['lyrics']]) for i in ids],batch_size=32,convert_to_numpy=True));moodnames=sorted(set(inp[i]['mood'] or 'unknown mood' for i in ids));pos={s:j for j,s in enumerate(moodnames)};me=unit(model.encode(moodnames,convert_to_numpy=True));ms=me@me.T;mood=[pos[inp[i]['mood'] or 'unknown mood'] for i in ids];genres=collections.defaultdict(list)
 for j,i in enumerate(ids):
  g=normalized(inp[i]['genre']).split(',')[0]
  if g:genres[g].append(j)
 lists=[[vocab[c] for c in m[i] if c and not vocab[c].startswith('<')] for i in ids];queries=unit(model.encode([', '.join(row) or 'no supported attributes' for row in lists],batch_size=32,convert_to_numpy=True));cache={};ranks=[];evaluated=[]
 for j,i in enumerate(ids):
  g=normalized(inp[i]['genre']).split(',')[0];key=(g,mood[j])
  if key not in cache:cache[key]=sorted(genres.get(g,[]),key=lambda k:(-float(ms[mood[j],mood[k]]),ids[k]))[:30]
  cand=[j]+[k for k in cache[key] if k!=j][:29]
  if len(cand)<5:continue
  evaluated.append(i)
  if not lists[j]:ranks.append(float('inf'));continue
  sim=target[cand]@queries[j];ranks.append(1+sum(sim[1:]>sim[0]+1e-8)+.5*sum(abs(sim[1:]-sim[0])<=1e-8))
 f=V/ds/'r31e-mrr-rows.json';f.write_text(json.dumps({'item_ids':evaluated,'reciprocal_ranks':[0 if np.isinf(r) else 1/r for r in ranks]})+'\n')
 p=V/ds/'r31e-mrr-summary.json';p.write_text(json.dumps({'dataset':ds,'mrr':float(np.mean(1/np.asarray(ranks))),'recall10':float(np.mean(np.asarray(ranks)<=10)),'n':len(ranks),'encoder':'all-mpnet-base-v2','same_original_candidate_rule':True,'elapsed_seconds':time.time()-t},indent=2)+'\n');print(ds,p.read_text(),flush=True)

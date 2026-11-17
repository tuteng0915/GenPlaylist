import collections, datetime, hashlib, json, re, time
from pathlib import Path
import numpy as np

R=Path('/storage_fast/ttu/GenPlaylist'); S=R/'outputs/cue-v2-stageA-20261112'; O=R/'outputs/cue-v2-r31d-20261113'; V=O/'v2'; D=O/'vocabulary-draft'; OUT=R/'outputs/cue-v2-r31e-20261113'
assert not OUT.exists(), 'Refusing to overwrite an existing version'
OUT.mkdir(parents=True)
def read(p): return json.loads(p.read_text())
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
vocab=read(V/'cue_vocab.json'); categories=read(V/'cue_categories.json'); emb=np.load(V/'cue_embeddings.npy'); vectors=np.load(D/'candidate-vectors.npy'); keys=[tuple(x) for x in read(D/'keys.json')]; pos={k:i for i,k in enumerate(keys)}; membership={int(k):v for k,v in read(V/'candidate-membership.json').items()}
inputs={d:read(S/d/'inputs.json') for d in ('mpd','m4a')};records={}; summ=np.zeros(vectors.shape[1],dtype=np.float64); count=0
for d in inputs:
 for i in inputs[d]:
  r=read(O/'full'/d/'candidates'/f'{i}.json');records[d,i]=r
  ix=[pos[c['category'],c['phrase']] for c in r['accepted']];summ+=vectors[ix].sum(axis=0);count+=len(ix)
globalmean=summ/count
cats=('sound','emotion','theme');quota=(2,2,4);threshold=.8
def tracklist(s):
 lines=[x.strip() for x in s.splitlines() if x.strip()]; matches=sum(bool(re.match(r'^.{2,80}\s[-–—]\s[\"“].{2,}',x)) for x in lines)
 return matches>=4 and matches>=len(lines)*.25
def instrumental(d,i,x):
 tags=x.get('tags') or []; tags=[tags] if isinstance(tags,str) else tags
 meta=' '.join(str(x.get(k) or '') for k in ('genre','mood','title','lyric_quality'))+' '+' '.join(tags)
 reasons=[]
 if not (x.get('lyrics') or '').strip():reasons.append('no_lyrics')
 if re.search(r'\binstrumental\b',meta,re.I):reasons.append('instrumental_metadata')
 if d=='mpd' and i=='106902':reasons.append('verified_green_onions_misattribution')
 if tracklist(x.get('lyrics') or ''):reasons.append('likely_release_tracklist')
 return reasons
def duplicate(a,b):
 x=vocab[a].casefold().strip();y=vocab[b].casefold().strip()
 substring=(x in y or y in x) and min(len(x),len(y))>=4
 return substring or float(emb[a]@emb[b])>threshold
def redundancy(row):
 if len(row)<2:return None
 return float(np.mean([emb[a]@emb[b] for k,a in enumerate(row) for b in row[k+1:]]))
reports={}
for d in inputs:
 old=read(V/d/'item2cues.json'); output={}; evidence={}; affected=[];removed=0;before=[];after=[];removed_dupes=0
 for i,x in inputs[d].items():
  prior=[k for k in old[i] if k];b=redundancy(prior)
  if b is not None:before.append(b)
  reasons=instrumental(d,i,x);unique={}
  for c in records[d,i]['accepted']:
   if reasons and c['category']=='theme':removed+=1;continue
   k=pos[c['category'],c['phrase']];j=membership[k];score=(float(vectors[k]@emb[j])-float(emb[j]@globalmean))*c['confidence']; row=dict(c,cue_id=j,canonical_phrase=vocab[j],mapping_score=score)
   if j not in unique or score>unique[j]['mapping_score']:unique[j]=row
  if reasons:affected.append({'item_id':i,'reasons':reasons,'previous_themes':sum(categories[k]=='theme' for k in prior)})
  remaining=list(unique.values());chosen=[];counts=collections.Counter()
  while remaining and len(chosen)<8:
   kept=[]
   for c in remaining:
    if any(duplicate(c['cue_id'],z['cue_id']) for z in chosen):removed_dupes+=1
    else:kept.append(c)
   remaining=kept
   if not remaining:break
   deficits=[c for c,q in zip(cats,quota) if counts[c]<q and any(z['category']==c for z in remaining)]; eligible=[z for z in remaining if not deficits or z['category'] in deficits]
   scores=[(.7*c['mapping_score']-.3*max([float(emb[c['cue_id']]@emb[z['cue_id']]) for z in chosen] or [0]),-c['cue_id'],c) for c in eligible];score,_,best=max(scores,key=lambda z:z[:2]);best=dict(best,selection_score=score);chosen.append(best);counts[best['category']]+=1;remaining=[c for c in remaining if c['cue_id']!=best['cue_id']]
  chosen.sort(key=lambda z:(cats.index(z['category']),-z['selection_score'],z['cue_id']));row=[c['cue_id'] for c in chosen];a=redundancy(row)
  if a is not None:after.append(a)
  assert all(not duplicate(z['cue_id'],w['cue_id']) for j,z in enumerate(chosen) for w in chosen[j+1:])
  output[i]=row+[0]*(8-len(row));evidence[i]=chosen
  if reasons:assert all(categories[k]!='theme' for k in row)
 write(OUT/d/'item2cues.json',output);write(OUT/d/'selection-evidence.json',evidence);write(OUT/d/'instrumental-or-tracklist.json',affected)
 reports[d]={'songs':len(output),'flagged':len(affected),'flagged_with_removed_prior_themes':sum(z['previous_themes']>0 for z in affected),'removed_candidate_themes':removed,'duplicate_candidates_skipped':removed_dupes,'mean_redundancy_before':float(np.mean(before)),'mean_redundancy_after':float(np.mean(after)),'redundancy_encoder':'existing MiniLM-L6 cue embeddings','nonempty_slots_before':sum(sum(bool(c) for c in row) for row in old.values()),'nonempty_slots_after':sum(sum(bool(c) for c in row) for row in output.values())}
for n in ('cue_vocab.json','cue_categories.json','cue_embeddings.npy'):
 (OUT/n).write_bytes((V/n).read_bytes())
paths=['cue_vocab.json','cue_categories.json','cue_embeddings.npy','mpd/item2cues.json','m4a/item2cues.json']
write(OUT/'repair-report.json',{'version':'cue-v2-r31e','started_by':'author requested targeted instrumental/near-duplicate repair; no extraction','created_sgt':datetime.datetime.now().astimezone().isoformat(),'threshold':threshold,'extra_rule':'conservative release-tracklist pattern and verified Green Onions ID; flagged separately','datasets':reports,'source_gate1_sha256':sha(O/'gate1.json'),'sha256':{n:sha(OUT/n) for n in paths},'warning':'Vocabulary and cluster naming retained from r31d; source gate1 numbers are before this repair.'})
print(json.dumps(reports,ensure_ascii=False),flush=True)

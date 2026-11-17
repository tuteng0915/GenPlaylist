"""Freeze a pilot-only vocabulary; compare pure relevance and legacy MMR at 8 cues."""
import collections
import numpy as np
from common import *

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def choose(candidates,emb,quota=(2,2,4),lam=.7):
 # Evidence threshold precedes this selector. Quotas can yield to available evidence.
 chosen=[];remaining=list(candidates);counts=collections.Counter()
 while remaining and len(chosen)<8:
  deficits=[c for c,q in zip(CATEGORIES,quota) if counts[c]<q and any(x['category']==c for x in remaining)]
  eligible=[x for x in remaining if not deficits or x['category'] in deficits]
  scored=[]
  for c in eligible:
   redundancy=max([float(emb[c['cue_id']]@emb[s['cue_id']]) for s in chosen] or [0.])
   score=lam*c['mapping_score']-(1-lam)*redundancy
   scored.append((score,-c['cue_id'],c))
  score,_,best=max(scored,key=lambda x:x[:2]);best=dict(best,selection_score=score);chosen.append(best);counts[best['category']]+=1;remaining=[x for x in remaining if x['cue_id']!=best['cue_id']]
 chosen.sort(key=lambda x:(CATEGORIES.index(x['category']),-x['selection_score'],x['cue_id']))
 return chosen

def legacy_mmr(ds,model,gate=False):
 inputs=json.loads((O/ds/'inputs.json').read_text());co=json.loads((O/ds/'cohort.json').read_text());x=paths(ds);oldv=json.loads((x['cues']/'cue_vocab.json').read_text())
 oldemb=unit(model.encode(oldv[1:],batch_size=128,convert_to_numpy=True));ids=sorted(set(r['condition_item'] for r in co['gate_rows'])) if gate else co['pilot'];texts=[' '.join(str(inputs[i][k]) if not isinstance(inputs[i][k],list) else ' '.join(inputs[i][k]) for k in ['title','genre','mood','tags','lyrics']) for i in ids]
 # Reproduce the ORIGINAL v1 song text exactly (raw metadata and 2000-character line dedup),
 # so this comparison changes selection only, not input cleaning or text length.
 meta=json.loads((x['data']/'catalog_metadata.json').read_text());texts=[]
 for i in ids:
  m=meta[i];lp=lyric_path(ds,i,m);raw=lp.read_text(errors='replace') if lp else ''
  seen=set();unique=[]
  for line in raw.splitlines():
   key=normalized(line)
   if key and key not in seen:unique.append(line.strip());seen.add(key)
  lyrics='\n'.join(unique)[:2000]
  texts.append(' '.join(p for p in [m.get('title'),m.get('genre'),m.get('mood'),m.get('lyric_excerpt'),*m.get('tags',[])] if p)+(' '+lyrics if lyrics else ''))
 se=unit(model.encode(texts,batch_size=64,convert_to_numpy=True));mmr={};reconstructed={};v1=json.loads((x['cues']/'item2cues.json').read_text())
 for i,v in zip(ids,se):
  rel=oldemb@v;order=np.lexsort((np.arange(len(rel)),-rel));reconstructed[i]=(order[:8]+1).tolist();candidates=order[:64].tolist();chosen=[]
  for _ in range(8):
   j=max(candidates,key=lambda j:(.7*float(rel[j])-.3*max([float(oldemb[j]@oldemb[k]) for k in chosen] or [0.]),-j));chosen.append(j);candidates.remove(j)
  mmr[i]=[j+1 for j in chosen]
 atomic(O/ds/('v1-mmr-gate-item2cues.json' if gate else 'v1-mmr-item2cues.json'),mmr);atomic(O/ds/('v1-gate-reconstruction-check.json' if gate else 'v1-reconstruction-check.json'),dict(exact_top8_rows=sum(reconstructed[i]==v1[i][:8] for i in ids),rows=len(ids),same_top8_set_rows=sum(set(reconstructed[i])==set(v1[i][:8]) for i in ids),order_only_differences=[dict(item_id=i,reconstructed=reconstructed[i],original=v1[i][:8]) for i in ids if reconstructed[i]!=v1[i][:8] and set(reconstructed[i])==set(v1[i][:8])],reconstructed=reconstructed,requirement='Must explain any mismatch before calling MMR a selection-only comparison'))

def main():
 from sentence_transformers import SentenceTransformer
 started=time.time();records={};pool_keys=set();allphrases=set()
 for ds in ['mpd','m4a']:
  co=json.loads((O/ds/'cohort.json').read_text())
  for i in co['extraction_ids']:
   p=O/ds/'candidates'/f'{i}.json';assert p.exists(),str(p);d=json.loads(p.read_text());records[(ds,i)]=d
   if i in co['vocab_pool']:pool_keys.add((ds,i));allphrases.update((c['category'],c['phrase']) for c in d['accepted'])
 keys=sorted(allphrases);assert keys
 model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda');model.max_seq_length=256
 vectors=unit(model.encode([p for c,p in keys],batch_size=128,convert_to_numpy=True,show_progress_bar=True));positions={k:i for i,k in enumerate(keys)}
 support=collections.defaultdict(set)
 for key in pool_keys:
  for c in records[key]['accepted']:support[(c['category'],c['phrase'])].add(key)
 # Deterministic bounded-radius clustering. Every member must be close to its seed;
 # unlike single linkage this cannot chain unrelated phrases through intermediates.
 clusters=[]
 for cat in CATEGORIES:
  pending=sorted([j for j,k in enumerate(keys) if k[0]==cat],key=lambda j:(-len(support[keys[j]]),keys[j][1]))
  while pending:
   seed=pending[0];sims=vectors[pending]@vectors[seed];members=[j for j,s in zip(pending,sims) if s>=.82]
   centroid=unit(vectors[members].mean(0,keepdims=True))[0];medoid=max(members,key=lambda j:(float(vectors[j]@centroid),keys[j][1]));songs=set().union(*(support[keys[j]] for j in members))
   if len(songs)>=5:clusters.append(dict(category=cat,phrase=keys[medoid][1],members=members,support=len(songs),medoid=medoid))
   used=set(members);pending=[j for j in pending if j not in used]
 budgets={'sound':300,'emotion':200,'theme':1547};selected=[]
 for cat in CATEGORIES:selected.extend(sorted([c for c in clusters if c['category']==cat],key=lambda c:(-c['support'],c['phrase']))[:budgets[cat]])
 selected.sort(key=lambda c:(CATEGORIES.index(c['category']),c['phrase']))
 vocab=['<unk>']+[c['phrase'] for c in selected];cats=['none']+[c['category'] for c in selected];active=len(vocab)-1
 while len(vocab)<2048:vocab.append(f'<pad_{len(vocab)}>');cats.append('none')
 ve=np.zeros((2048,vectors.shape[1]),dtype=np.float32)
 for j,c in enumerate(selected,1):ve[j]=vectors[c['medoid']]
 # Average over candidate occurrences, not just unique phrase strings.
 pool_occurrences=[positions[(c['category'],c['phrase'])] for key in sorted(pool_keys) for c in records[key]['accepted']]
 global_candidate_mean=vectors[pool_occurrences].mean(0)
 mapping={};selection_records={};unmapped=0
 for key,d in records.items():
  cand=d['accepted'];cv=unit(model.encode([c['phrase'] for c in cand],convert_to_numpy=True)) if cand else np.empty((0,ve.shape[1]));mapped={}
  for c,v in zip(cand,cv):
   ids=np.array([j for j in range(1,active+1) if cats[j]==c['category']],dtype=np.int64);sims=ve[ids]@v
   valid=ids[sims>=.78]
   if not len(valid):unmapped+=1;continue
   scores=ve[valid]@v-ve[valid]@global_candidate_mean;j=int(valid[np.argmax(scores)]);score=float(max(scores))*float(c['confidence'])
   rec=dict(c,cue_id=j,canonical_phrase=vocab[j],mapping_score=score,cosine=float(ve[j]@v))
   if j not in mapped or score>mapped[j]['mapping_score']:mapped[j]=rec
  chosen=choose(list(mapped.values()),ve);mapping[key]=[c['cue_id'] for c in chosen]+[0]*(8-len(chosen));selection_records[key]=chosen
 immutable(O/'v2/cue_vocab.json',vocab);immutable(O/'v2/cue_categories.json',cats)
 atomic(O/'v2/vocabulary-support.json',[dict(cue_id=j,category=c['category'],phrase=c['phrase'],songs=c['support'],member_phrases=[keys[k][1] for k in c['members']]) for j,c in enumerate(selected,1)])
 np.save(O/'v2/cue_embeddings.npy',ve)
 for ds in ['mpd','m4a']:
  atomic(O/ds/'v2-item2cues.json',{i:v for (s,i),v in mapping.items() if s==ds});atomic(O/ds/'v2-selection-evidence.json',{i:v for (s,i),v in selection_records.items() if s==ds})
  legacy_mmr(ds,model)
 config=dict(scope='Stage A pilot vocabulary, not formal full rebuild',active_real_cues=active,padding_slots=2047-active,pilot_padding='Unused reserved IDs only; never selected. Small pilot may not support2047 concepts with>=5 songs.',minimum_distinct_song_support=5,cluster_cosine=.82,mapping_cosine=.78,mapping_score='cosine(candidate,canonical) minus mean cosine(canonical,all vocabulary-pool candidate occurrences), times confidence',quota=[2,2,4],mmr_lambda=.7,unmapped_candidates=unmapped,elapsed_seconds=time.time()-started,stage_B_not_started=True)
 atomic(O/'build-report.json',config);print(json.dumps(config),flush=True)

if __name__=='__main__':main()

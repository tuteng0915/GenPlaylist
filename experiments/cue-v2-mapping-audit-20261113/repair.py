"""Stage A r2: all3544 songs, support2, explicit rare-cluster merge, own-cluster mapping.
No audio generation. No writes to original Stage A artifacts. Coverage gate is mandatory.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,time,json,collections,hashlib
from pathlib import Path
import numpy as np
from audit import cluster,unit,CATS,BUDGET
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';A=R/'outputs/cue-v2-mapping-audit-20261113';O=R/'outputs/cue-v2-repair-r2-20261113';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from build import choose
from common import sha

def save(n,d):
 p=O/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
 started=time.time();z=np.load(A/'phrase-embeddings.npz');keys=[tuple(k) for k in json.loads(str(z['keys']))];vectors=z['vectors'];pos={k:j for j,k in enumerate(keys)};records={};cohorts={};support=collections.defaultdict(set)
 for ds in ['mpd','m4a']:
  co=cohorts[ds]=json.loads((S/ds/'cohort.json').read_text())
  for i in co['extraction_ids']:
   d=records[(ds,i)]=json.loads((S/ds/'candidates'/f'{i}.json').read_text())
   for c in d['accepted']:support[(c['category'],c['phrase'])].add((ds,i))
 assert len(records)==3544
 fine=cluster(keys,vectors,support,.82,.78);shared=[c for c in fine if c['support']>=2];rare=[c for c in fine if c['support']<2];merges=[]
 # Keep fine shared meanings. Broaden only unresolved rare clusters, without changing category.
 # Similarity is a construction guard; assigned members will NOT be filtered again later.
 for cat in CATS:
  pending=sorted([c for c in rare if c['category']==cat],key=lambda c:c['phrase'])
  while pending:
   seed=pending[0];group=[seed];members=list(seed['members']);unused=[]
   for c in pending[1:]:
    # All members must remain close to the seed representative: no single-link chains.
    if float((vectors[c['members']]@vectors[seed['medoid']]).min())>=.65:
     group.append(c);members.extend(c['members'])
    else:unused.append(c)
   songs=set().union(*(support[keys[j]] for j in members));center=unit(vectors[members].mean(0,keepdims=True))[0];eligible=[j for j in members if float((vectors[members]@vectors[j]).min())>=.65-1e-7];assert eligible
   medoid=max(eligible,key=lambda j:(float(vectors[j]@center),keys[j][1]));parent=dict(category=cat,phrase=keys[medoid][1],members=members,support=len(songs),medoid=medoid,seed=seed['medoid'],min_member_cosine=float((vectors[members]@vectors[medoid]).min()),construction='rare semantic coarsening, requires semantic audit; not guaranteed broader merely by cosine')
   if len(songs)>=2:
    shared.append(parent);merges.append(dict(parent=parent['phrase'],children=[c['phrase'] for c in group],songs=len(songs),category=cat))
   else:
    # Try a compatible already shared parent; keep unresolved candidates explicitly if unsafe.
    targets=[c for c in shared if c['category']==cat and float((vectors[members]@vectors[c['medoid']]).min())>=.65]
    if targets:
     target=max(targets,key=lambda c:float((vectors[members]@vectors[c['medoid']]).mean()));target['members']+=members;target['min_member_cosine']=float((vectors[target['members']]@vectors[target['medoid']]).min());target['support']=len(set().union(*(support[keys[j]] for j in target['members'])));merges.append(dict(parent=target['phrase'],children=[c['phrase'] for c in group],songs=target['support'],category=cat))
   pending=unused
 # Fixed 2047 real IDs. Unused slots are padding; quotas are soft, never grounds for deleting evidence.
 shared.sort(key=lambda c:(-c['support'],CATS.index(c['category']),c['phrase']));budget_dropped=shared[2047:];shared=shared[:2047];shared.sort(key=lambda c:(CATS.index(c['category']),c['phrase']))
 vocab=['<unk>']+[c['phrase'] for c in shared];cats=['none']+[c['category'] for c in shared];member_to_id={}
 for j,c in enumerate(shared,1):
  for member in c['members']:
   assert member not in member_to_id,'Clusters overlap';member_to_id[member]=j
 ve=np.vstack([np.zeros(vectors.shape[1]),vectors[[c['medoid'] for c in shared]]]);poolocc=[pos[(c['category'],c['phrase'])] for key in sorted(records) for c in records[key]['accepted']];mean=vectors[poolocc].mean(0);out={};mapping={};selection={};unresolved={}
 for ds in ['mpd','m4a']:
  mapping[ds]={};selection[ds]={};unresolved[ds]={};funnel=collections.Counter();quality=collections.Counter()
  for (dataset,i),d in records.items():
   if dataset!=ds:continue
   quality[d['input']['lyric_quality']]+=1;funnel['songs']+=1;funnel['raw_candidates']+=len(d['accepted'])+len(d['rejected']);funnel['after_evidence_filter']+=len(d['accepted']);mapped={};miss=[]
   for c in d['accepted']:
    k=pos[(c['category'],c['phrase'])];j=member_to_id.get(k)
    if j is None:miss.append(dict(c,status='no_safe_shared_parent'));continue
    sim=float(vectors[k]@ve[j]);score=(sim-float(ve[j]@mean))*c['confidence'];r=dict(c,cue_id=j,canonical_phrase=vocab[j],mapping_score=score,cosine=sim,mapping='own-cluster membership; no posthoc threshold');funnel['mapped_candidates']+=1
    if j not in mapped or score>mapped[j]['mapping_score']:mapped[j]=r
   funnel['unique_mapped']+=len(mapped);chosen=choose(list(mapped.values()),ve);mapping[ds][i]=[c['cue_id'] for c in chosen]+[0]*(8-len(chosen));selection[ds][i]=chosen;unresolved[ds][i]=miss;funnel['selected_slots']+=len(chosen)
  groups={}
  for group,ids in [('all',cohorts[ds]['extraction_ids']),('pilot',cohorts[ds]['pilot'])]:
   eligible=[i for i in ids if records[(ds,i)]['input']['lyrics']];n=sum(sum(j!=0 for j in mapping[ds][i]) for i in eligible);ceiling=sum(min(8,len(records[(ds,i)]['accepted'])) for i in eligible)
   groups[group]={'songs':len(ids),'songs_with_usable_lyrics':len(eligible),'nonempty_slots_on_lyric_songs':n,'possible_slots_on_lyric_songs':8*len(eligible),'coverage_on_lyric_songs':n/max(1,8*len(eligible)),'candidate_count_upper_bound_slots':ceiling,'candidate_count_upper_bound_coverage':ceiling/max(1,8*len(eligible)),'passed90percent':n>=.9*8*len(eligible)}
  out[ds]={'funnel':dict(funnel),'quality':dict(quality),'coverage':groups};save(ds+'/item2cues.json',mapping[ds]);save(ds+'/selection-evidence.json',selection[ds]);save(ds+'/unresolved-candidates.json',unresolved[ds])
 active=len(vocab)-1
 while len(vocab)<2048:vocab.append('<pad_'+str(len(vocab))+'>');cats.append('none')
 save('cue_vocab.json',vocab);save('cue_categories.json',cats);np.save(O/'cue_embeddings.npy',ve);save('cluster-support.json',shared);save('coarsening-records.json',merges)
 save('diagnostic-funnel.json',dict(datasets=out,active_vocab=active,fine_clusters=len(fine),fine_shared_clusters=sum(c['support']>=2 for c in fine),coarsening_groups=len(merges),budget_dropped_clusters=len(budget_dropped),pool_songs=len(records),minimum_support=2,own_cluster_mapping=True,coverage_gate_passed=all(out[d]['coverage']['all']['passed90percent'] for d in out),audio_authorized_only_after_coverage_gate=True,audio_started=False,stage_B_started=False,semantic_warning='Coarse mapping remains provisional until audit; coverage is not faithfulness.',elapsed_seconds=time.time()-started,embedding_source_sha256=sha(A/'phrase-embeddings.npz')))
 print(json.dumps(out,indent=2));print('REPAIR_DIAGNOSTICS_DONE',time.time()-started,flush=True)
if __name__=='__main__':main()

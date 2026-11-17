"""Read-only Stage A diagnosis. CPU-only; all new artifacts isolated from live jobs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,time,collections,hashlib
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';O=R/'outputs/cue-v2-mapping-audit-20261113';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from build import choose
CATS=['sound','emotion','theme'];BUDGET={'sound':300,'emotion':200,'theme':1547}
def dump(name,data):(O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cluster(keys,vec,support,threshold,representative_floor=None):
 out=[]
 for cat in CATS:
  pending=sorted([j for j,k in enumerate(keys) if k[0]==cat],key=lambda j:(-len(support[keys[j]]),keys[j][1]))
  while pending:
   seed=pending[0];sim=vec[pending]@vec[seed];members=[j for j,s in zip(pending,sim) if s>=threshold]
   center=unit(vec[members].mean(0,keepdims=True))[0];eligible=members
   if representative_floor is not None:
    eligible=[j for j in members if float((vec[members]@vec[j]).min())>=representative_floor-1e-7]
    assert eligible, 'Seed must satisfy the representative floor'
   medoid=max(eligible,key=lambda j:(float(vec[j]@center),keys[j][1]));songs=set().union(*(support[keys[j]] for j in members))
   out.append(dict(category=cat,phrase=keys[medoid][1],members=members,support=len(songs),medoid=medoid,seed=seed,min_member_cosine=float((vec[members]@vec[medoid]).min())))
   used=set(members);pending=[j for j in pending if j not in used]
 return out

def main():
 started=time.time();records={};cohorts={};pool=set();sourcehash={};allkeys=set();poolkeys=set()
 for ds in ['mpd','m4a']:
  co=cohorts[ds]=json.loads((S/ds/'cohort.json').read_text());pool.update((ds,i) for i in co['vocab_pool'])
  for i in co['extraction_ids']:
   p=S/ds/'candidates'/f'{i}.json';d=records[(ds,i)]=json.loads(p.read_text());sourcehash[str(p)]=sha(p)
   allkeys.update((c['category'],c['phrase']) for c in d['accepted'])
   if i in co['vocab_pool']:poolkeys.update((c['category'],c['phrase']) for c in d['accepted'])
 keys=sorted(poolkeys);allkeys=sorted(allkeys);allpos={k:j for j,k in enumerate(allkeys)};pos={k:j for j,k in enumerate(keys)}
 cache=O/'phrase-embeddings.npz'
 if cache.exists():
  z=np.load(cache);assert json.loads(str(z['keys']))==[list(k) for k in allkeys];av=z['vectors']
 else:
  import torch
  torch.set_num_threads(1)
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);model.max_seq_length=256
  av=unit(model.encode([k[1] for k in allkeys],batch_size=64,convert_to_numpy=True,show_progress_bar=False));np.savez(cache,keys=json.dumps(allkeys),vectors=av)
 vec=av[[allpos[k] for k in keys]];support=collections.defaultdict(set)
 for key in pool:
  for c in records[key]['accepted']:support[(c['category'],c['phrase'])].add(key)
 mean=vec[[pos[(c['category'],c['phrase'])] for key in sorted(pool) for c in records[key]['accepted']]].mean(0)
 dump('input-manifest.json',dict(source_hashes=sourcehash,pool_songs=len(pool),accepted_phrases=len(allkeys),pool_phrases=len(keys),encoder='all-MiniLM-L6-v2',device='cpu',threads=1))
 base=cluster(keys,vec,support,.82)
 dump('cluster-funnel.json',dict(total_clusters=len(base),clusters_by_category=dict(collections.Counter(c['category'] for c in base)),support_histogram=dict(collections.Counter(c['support'] for c in base)),kept_min5=sum(c['support']>=5 for c in base),dropped_min5=sum(c['support']<5 for c in base),retained_cluster_members_below_mapping_threshold=sum(sum(float(vec[j]@vec[c['medoid']])<.78 for j in c['members']) for c in base if c['support']>=5),original_representatives_shifted_from_seed=sum(c['seed']!=c['medoid'] for c in base)))
 configs=[('original',.82,5,.78,False),('representative_fix',.82,5,.78,True),('map_only_070',.82,5,.70,False),('map_only_065',.82,5,.65,False),('support2_diagnostic',.82,2,.78,True),('coarse072',.72,5,.72,True),('coarse065',.65,5,.65,True)]
 result={};auditrows={};oldmap={ds:json.loads((S/ds/'v2-item2cues.json').read_text()) for ds in cohorts};oldvoc=json.loads((S/'v2/cue_vocab.json').read_text())
 heldout={ds:sorted({i for r in co['gate_rows'] for i in r['targets']} - set(co['vocab_pool']))[:40] for ds,co in cohorts.items()}
 dump('evaluation-cohorts.json',dict(pilot={ds:co['pilot'] for ds,co in cohorts.items()},heldout=heldout,heldout_rule='first40 sorted gate-target IDs outside vocabulary pool; fixed before examining results'))
 for name,ct,minimum,mt,fix in configs:
  clusters=base if ct==.82 and not fix else cluster(keys,vec,support,ct,mt if fix else None)
  selected=[]
  for cat in CATS:selected.extend(sorted([c for c in clusters if c['category']==cat and c['support']>=minimum],key=lambda c:(-c['support'],c['phrase']))[:BUDGET[cat]])
  selected.sort(key=lambda c:(CATS.index(c['category']),c['phrase']));vs=np.vstack([np.zeros(vec.shape[1]),vec[[c['medoid'] for c in selected]]]);vocab=['<unk>']+[c['phrase'] for c in selected];cats=['none']+[c['category'] for c in selected]
  r=dict(cluster_threshold=ct,min_song_support=minimum,mapping_threshold=mt,representative_fix=fix,vocabulary_size=len(selected),vocabulary_by_category=dict(collections.Counter(c['category'] for c in selected)),datasets={});rows={}
  for ds,co in cohorts.items():
   rows[ds]={};r['datasets'][ds]={}
   for group,ids in [('pilot',co['pilot']),('heldout',heldout[ds])]:
    cnt=collections.Counter();catcnt=collections.Counter();maxs=[];chosenall=[];oldexact=0
    for i in ids:
     rec=records[(ds,i)];cand=rec['accepted'];cnt['songs']+=1;cnt['accepted']+=len(cand);cnt['raw']+=len(cand)+len(rec['rejected']);cnt['songs_with_accepted']+=bool(cand);mapped={};miss=[]
     for c in cand:
      v=av[allpos[(c['category'],c['phrase'])]];eligible=np.array([j for j in range(1,len(vocab)) if cats[j]==c['category']],dtype=int)
      if not len(eligible):cnt['no_category_vocab']+=1;miss.append(c);continue
      sim=vs[eligible]@v;maxs.append(float(sim.max()));valid=eligible[sim>=mt]
      if not len(valid):cnt['below_mapping']+=1;miss.append(c);continue
      cnt['mapped_candidates']+=1;scores=vs[valid]@v-vs[valid]@mean;j=int(valid[np.argmax(scores)]);score=float(max(scores))*c['confidence'];new=dict(c,cue_id=j,canonical_phrase=vocab[j],mapping_score=score,cosine=float(vs[j]@v))
      if j not in mapped or score>mapped[j]['mapping_score']:mapped[j]=new
     cnt['unique_mapped']+=len(mapped)
     chosen=choose(list(mapped.values()),vs);cnt['selected']+=len(chosen);cnt['songs_with_selected']+=bool(chosen);cnt['beyond8_removed']+=max(0,len(mapped)-8)
     catcnt.update(c['category'] for c in chosen);chosenall.append([c['cue_id'] for c in chosen]);rows[ds][i]=dict(input=rec['input'],accepted=cand,chosen=chosen,unmapped=miss)
     if [vocab[c['cue_id']] for c in chosen]==[oldvoc[j] for j in oldmap[ds][i] if j]:oldexact+=1
    cnt['merge_collisions']=cnt['mapped_candidates']-cnt['unique_mapped'];occ=collections.Counter(v for row in chosenall for v in row);rep=[]
    for row in chosenall:
     if len(row)>1:rep.append(float((vs[row]@vs[row].T)[np.triu_indices(len(row),1)].mean()))
    r['datasets'][ds][group]=dict(cnt,empty_slots=8*len(ids)-cnt['selected'],nonempty_rate=cnt['selected']/(8*len(ids)),selected_categories=dict(catcnt),mapping_max_cosine_quantiles=np.quantile(maxs,[0,.25,.5,.75,1]).tolist() if maxs else [],top10_share=sum(v for _,v in occ.most_common(10))/max(1,sum(occ.values())),mean_pairwise_cosine=float(np.mean(rep)) if rep else None,exact_original_rows=oldexact)
  result[name]=r;dump(name+'-mapping.json',rows);dump(name+'-vocabulary.json',dict(vocab=vocab,categories=cats,support=[{k:v for k,v in c.items() if k!='members'} for c in selected]));print(name,{ds:r['datasets'][ds]['pilot']['selected'] for ds in cohorts},flush=True)
 dump('comparison.json',dict(variants=result,elapsed_seconds=time.time()-started,stage_B_started=False,live_stageA_changed=False))
 print('AUDIT_DONE',time.time()-started,flush=True)
if __name__=='__main__':main()

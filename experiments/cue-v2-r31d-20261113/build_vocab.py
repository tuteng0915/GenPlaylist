"""Full-catalog draft clusters; every supported candidate keeps its own membership.
Canonical names/semantic validation are a separate stage. Nothing trains here.
"""
import collections,json,os,sys,time
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-r31d-20261113';S=R/'outputs/cue-v2-stageA-20261112'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import atomic,immutable,sha,CATEGORIES
from build import unit

def main():
    started=time.time();records={};support=collections.defaultdict(set);hashes={}
    for ds in ['mpd','m4a']:
        inp=json.loads((S/ds/'inputs.json').read_text());base=O/'full'/ds
        for k in range(4):assert (base/f'complete-{k}.json').exists(),(ds,k)
        for i in sorted(inp):
            p=base/'candidates'/f'{i}.json';d=json.loads(p.read_text());assert d['input']==inp[i] and d['config_sha256']==sha(base/'config.json');records[ds,i]=d;hashes[f'{ds}/{i}']=sha(p)
            for c in d['accepted']:support[c['category'],c['phrase']].add((ds,i))
    assert len(records)==16069
    keys=sorted(support);positions={k:i for i,k in enumerate(keys)}
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    import torch
    torch.set_num_threads(4)
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda',local_files_only=True);model.max_seq_length=256
    vectors=unit(model.encode([p for c,p in keys],batch_size=128,convert_to_numpy=True,show_progress_bar=True))
    budgets={'sound':300,'emotion':200,'theme':1547};groups=[];merge_log=[]
    # Fixed construction, not selected by downstream MRR. Budgets are ceilings;
    # rare clusters are merged rather than pruned. No mapping cosine cutoff.
    for cat in CATEGORIES:
        ids=np.array([i for i,k in enumerate(keys) if k[0]==cat]);n=min(budgets[cat],len(ids))
        if not n:continue
        km=MiniBatchKMeans(n_clusters=n,random_state=42,n_init=10,batch_size=4096,max_iter=200,reassignment_ratio=0.)
        labels=km.fit_predict(vectors[ids]);local=[]
        for label in sorted(set(labels.tolist())):
            members=ids[labels==label].tolist();songs=set().union(*(support[keys[i]] for i in members));center=unit(vectors[members].mean(0,keepdims=True))[0]
            local.append(dict(category=cat,members=members,songs=songs,center=center,coarsened=False))
        while any(len(g['songs'])<3 for g in local):
            assert len(local)>1,(cat,'cannot achieve support3 without discarding')
            a=min((j for j,g in enumerate(local) if len(g['songs'])<3),key=lambda j:(len(local[j]['songs']),min(local[j]['members'])))
            b=max((j for j in range(len(local)) if j!=a),key=lambda j:(float(local[a]['center']@local[j]['center']),-min(local[j]['members'])))
            ga,gb=local[a],local[b];merge_log.append(dict(category=cat,source_members=ga['members'],target_members=gb['members'][:],cosine=float(ga['center']@gb['center']),source_songs=len(ga['songs']),target_songs=len(gb['songs'])))
            gb['members']+=ga['members'];gb['songs']|=ga['songs'];gb['center']=unit(vectors[gb['members']].mean(0,keepdims=True))[0];gb['coarsened']=True;local.pop(a)
        for g in local:
            members=sorted(g['members']);medoid=max(members,key=lambda j:(float(vectors[j]@g['center']),keys[j][1]));minimum=float((vectors[members]@vectors[medoid]).min())
            groups.append(dict(category=cat,members=members,medoid=medoid,suggested_name=keys[medoid][1],support=len(g['songs']),coarsened=g['coarsened'],min_member_cosine=minimum,requires_broader_name=g['coarsened'] or minimum<.82))
    groups.sort(key=lambda g:(CATEGORIES.index(g['category']),g['suggested_name'],g['medoid']))
    membership={i:j+1 for j,g in enumerate(groups) for i in g['members']};assert set(membership)==set(range(len(keys)));assert sum(len(g['members']) for g in groups)==len(keys);assert len(groups)<=2047 and all(g['support']>=3 for g in groups)
    out=O/'vocabulary-draft';out.mkdir(exist_ok=True)
    np.save(out/'candidate-vectors.npy',vectors);immutable(out/'keys.json',keys);immutable(out/'clusters.json',groups);atomic(out/'rare-merge-log.json',merge_log);atomic(out/'source-hashes.json',hashes)
    jobs=[]
    for j,g in enumerate(groups,1):
        phrases=[keys[k][1] for k in g['members']]
        if not g['requires_broader_name']:continue
        # Every member is presented. Oversized groups need explicit hierarchical
        # naming, rather than silently truncating evidence to a convenient sample.
        jobs.append(dict(cue_id=j,category=g['category'],suggested_name=g['suggested_name'],member_phrases=phrases,support=g['support']))
    atomic(out/'naming-requests.json',jobs)
    atomic(out/'complete.json',dict(songs=len(records),accepted_occurrences=sum(len(d['accepted']) for d in records.values()),unique_candidates=len(keys),clusters=len(groups),minimum_support=3,unmapped_candidates=0,dropped_candidates=0,rare_merges=len(merge_log),naming_requests=len(jobs),requires_naming_before_selection=True,method='category MiniBatchKMeans fixedseed42, budgets300/200/1547; rare same-category nearest-centroid merges; own membership; no postmapping threshold',elapsed_seconds=time.time()-started,script_sha256=sha(Path(__file__))))
    print('VOCAB_DRAFT_READY',len(groups),len(jobs),time.time()-started,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'vocabulary-draft/failure.json',dict(traceback=traceback.format_exc()));raise

"""Post-vocabulary coverage and matched all-catalog diagnostics. A failed gate never trains."""
import collections,json,sys,time
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-r31d-20261113';S=R/'outputs/cue-v2-stageA-20261112';D=O/'vocabulary-draft';V=O/'v2'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import atomic,immutable,sha,paths,CATEGORIES,normalized
from build import unit,choose

def concentration(mapping):
    occ=collections.Counter(c for row in mapping.values() for c in row if c);songs=collections.Counter(c for row in mapping.values() for c in set(row) if c);n=sum(occ.values());pr=np.array(list(occ.values()),float)/max(1,n)
    return dict(max_song_fraction=max(songs.values(),default=0)/len(mapping),top10_occurrence_share=sum(n for _,n in occ.most_common(10))/max(1,n),entropy_bits=float(-(pr*np.log2(pr)).sum()),nonempty_slots=n)

def main():
    started=time.time();names=json.loads((D/'naming-complete.json').read_text());groups=json.loads((D/'clusters.json').read_text());keys=[tuple(k) for k in json.loads((D/'keys.json').read_text())];vectors=np.load(D/'candidate-vectors.npy');pos={k:i for i,k in enumerate(keys)};records={};inputs={};sourcehash=json.loads((D/'source-hashes.json').read_text())
    for ds in ['mpd','m4a']:
        inputs[ds]=json.loads((S/ds/'inputs.json').read_text())
        for i in inputs[ds]:
            p=O/'full'/ds/'candidates'/f'{i}.json';assert sha(p)==sourcehash[f'{ds}/{i}'];records[ds,i]=json.loads(p.read_text())
    canonical=[]
    for j,g in enumerate(groups,1):
        p=D/'names'/f'{j}.json';name=json.loads(p.read_text())['result']['name'] if p.exists() else g['suggested_name']
        canonical.append((g['category'],name or g['suggested_name']))
    # Equal canonical labels in the same category are ONE cue, never duplicate IDs.
    ckeys=sorted(set(canonical),key=lambda k:(CATEGORIES.index(k[0]),k[1]));cp={k:i+1 for i,k in enumerate(ckeys)};membership={i:cp[canonical[j]] for j,g in enumerate(groups) for i in g['members']};assert len(membership)==len(keys)
    vocab=['<unk>']+[p for c,p in ckeys];categories=['none']+[c for c,p in ckeys]
    while len(vocab)<2048:vocab.append(f'<pad_{len(vocab)}>');categories.append('none')
    assert len(vocab)==2048
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    mini=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cuda',local_files_only=True);mini.max_seq_length=256
    ve=np.zeros((2048,384),np.float32);ve[1:len(ckeys)+1]=unit(mini.encode([p for c,p in ckeys],batch_size=128,convert_to_numpy=True))
    occ=[pos[c['category'],c['phrase']] for d in records.values() for c in d['accepted']];globalmean=vectors[occ].mean(0);maps={};report={};supports=collections.defaultdict(set)
    for ds in ['mpd','m4a']:
        mapping={};selection={};funnel=collections.Counter();cover={}
        for i in inputs[ds]:
            d=records[ds,i];unique={};funnel['candidates_before_filter']+=len(d['accepted'])+len(d['rejected']);funnel['after_evidence_filter']+=len(d['accepted'])
            for c in d['accepted']:
                k=pos[c['category'],c['phrase']];j=membership[k];sim=float(vectors[k]@ve[j]);score=(sim-float(ve[j]@globalmean))*c['confidence'];row=dict(c,cue_id=j,canonical_phrase=vocab[j],mapping_score=score,cosine=sim,mapping='own-cluster membership, no posthoc rejection');supports[j].add((ds,i));funnel['mapped_candidates']+=1
                if j not in unique or score>unique[j]['mapping_score']:unique[j]=row
            funnel['unique_mapped_candidates']+=len(unique);selected=choose(list(unique.values()),ve);selection[i]=selected;mapping[i]=[r['cue_id'] for r in selected]+[0]*(8-len(selected));funnel['selected_slots']+=len(selected)
        assert funnel['after_evidence_filter']==funnel['mapped_candidates']
        for label,ids in [('with_lyrics',[i for i,d in inputs[ds].items() if d['lyrics']]),('without_lyrics',[i for i,d in inputs[ds].items() if not d['lyrics']])]:
            n=[sum(c!=0 for c in mapping[i]) for i in ids];cover[label]=dict(songs=len(ids),selected_slots=sum(n),possible_slots=8*len(ids),coverage=sum(n)/max(1,8*len(ids)),min_slots=min(n,default=0),slot_histogram=dict(collections.Counter(n)))
        maps[ds]=mapping;report[ds]=dict(funnel=dict(funnel),postmapping_coverage=cover)
        atomic(V/ds/'item2cues.json',mapping);atomic(V/ds/'selection-evidence.json',selection)
    assert all(len(s)>=3 for s in supports.values())
    immutable(V/'cue_vocab.json',vocab);immutable(V/'cue_categories.json',categories);np.save(V/'cue_embeddings.npy',ve);atomic(V/'canonical-support.json',{str(j):len(s) for j,s in supports.items()});atomic(V/'candidate-membership.json',{str(k):v for k,v in membership.items()})
    del mini
    torch.cuda.empty_cache()
    model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device='cuda',local_files_only=True)
    for ds in ['mpd','m4a']:
        inp=inputs[ds];ids=sorted(inp);oldv=json.loads((paths(ds)['cues']/'cue_vocab.json').read_text());v1=json.loads((paths(ds)['cues']/'item2cues.json').read_text());mmr=json.loads((R/'outputs/cue-v2-followup-20261113'/ds/'catalog-mmr-item2cues.json').read_text());armmaps={'v1':{i:v1[i][:8] for i in ids},'mmr':{i:mmr[i][:8] for i in ids},'r31':maps[ds]}
        assert set(ids)==set(v1)==set(mmr)
        target=unit(model.encode([' '.join([inp[i]['genre'],' '.join(inp[i]['tags']),inp[i]['mood'],inp[i]['lyrics']]) for i in ids],batch_size=64,convert_to_numpy=True));moodnames=sorted(set(inp[i]['mood'] or 'unknown mood' for i in ids));moodpos={s:j for j,s in enumerate(moodnames)};me=unit(model.encode(moodnames,convert_to_numpy=True));ms=me@me.T;moodix=[moodpos[inp[i]['mood'] or 'unknown mood'] for i in ids];genres=collections.defaultdict(list)
        for j,i in enumerate(ids):
            g=normalized(inp[i]['genre']).split(',')[0]
            if g:genres[g].append(j)
        cache={};candidates=[]
        for j,i in enumerate(ids):
            g=normalized(inp[i]['genre']).split(',')[0];key=(g,moodix[j])
            if key not in cache:cache[key]=sorted(genres.get(g,[]),key=lambda k:(-float(ms[moodix[j],moodix[k]]),ids[k]))[:30]
            candidates.append([j]+[k for k in cache[key] if k!=j][:29])
        report[ds]['diagnostic_protocol']=dict(scope='entire catalog, same candidate sets for every arm',candidate_rule='same primary genre, nearest metadata mood, maximum30 including source; omit groups<5',encoder='all-mpnet-base-v2',target_text='genre tags mood cleaned lyrics; encoder default truncation as original StageA MRR',encoder_max_seq_length=model.max_seq_length)
        metrics={}
        for arm,mapping in armmaps.items():
            vv=vocab if arm=='r31' else oldv;lists=[[vv[c] for c in mapping[i] if c and not vv[c].startswith('<')] for i in ids];queries=unit(model.encode([', '.join(row) or 'no supported attributes' for row in lists],batch_size=64,convert_to_numpy=True));ranks=[];evaluated=[]
            strings=sorted({p for row in lists for p in row});sp={p:j for j,p in enumerate(strings)};ee=unit(model.encode(strings,batch_size=128,convert_to_numpy=True));red=[];maxred=[]
            for j,cand in enumerate(candidates):
                row=lists[j]
                if len(row)>1:
                    e=ee[[sp[p] for p in row]];sim=(e@e.T)[np.triu_indices(len(row),1)];red.append(float(sim.mean()));maxred.append(float(sim.max()))
                if len(cand)<5:continue
                evaluated.append(ids[j])
                if not row:ranks.append(float('inf'));continue
                sim=target[cand]@queries[j];ranks.append(1+sum(sim[1:]>sim[0]+1e-8)+.5*sum(abs(sim[1:]-sim[0])<=1e-8))
            metrics[arm]=dict(concentration=concentration(mapping),retrieval_n=len(ranks),mrr=float(np.mean(1/np.asarray(ranks))) if ranks else None,recall10=float(np.mean(np.asarray(ranks)<=10)) if ranks else None,mean_pairwise_cosine=float(np.mean(red)) if red else None,mean_max_pairwise_cosine=float(np.mean(maxred)) if maxred else None)
            atomic(V/ds/(arm+'-mrr-rows.json'),dict(item_ids=evaluated,reciprocal_ranks=[0 if np.isinf(r) else 1/r for r in ranks]))
        report[ds]['arms']=metrics
        conditions=dict(lyric_coverage_at_least85=report[ds]['postmapping_coverage']['with_lyrics']['coverage']>=.85,mrr_at_least_v1=metrics['r31']['mrr'] is not None and metrics['r31']['mrr']>=metrics['v1']['mrr'],max_cue_fraction_below_mmr=metrics['r31']['concentration']['max_song_fraction']<metrics['mmr']['concentration']['max_song_fraction'],top10_share_below_mmr=metrics['r31']['concentration']['top10_occurrence_share']<metrics['mmr']['concentration']['top10_occurrence_share'],entropy_above_mmr=metrics['r31']['concentration']['entropy_bits']>metrics['mmr']['concentration']['entropy_bits'])
        report[ds]['gate1_conditions']=conditions;report[ds]['passed']=all(conditions.values());atomic(V/ds/'diagnostics.json',report[ds])
    result=dict(passed=bool(names['passed'] and all(r['passed'] for r in report.values())),datasets=report,naming_passed=names['passed'],incoherent_clusters=names['incoherent'],active_cues=len(ckeys),padding_ids=2047-len(ckeys),mapped_coverage_is_postclustering=True,no_lyric_minimum_waived=True,personalization_diagnostic='pending separate941-context report; no claim of personalization improvement',training_started=False,elapsed_seconds=time.time()-started)
    atomic(O/'gate1.json',result);print('GATE1',result['passed'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'mapping-failure.json',dict(traceback=traceback.format_exc()));raise

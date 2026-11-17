"""Frozen-prediction cue audit: pooled multiset F1 vs optimal item matching."""
import argparse,gzip,hashlib,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PERMS=np.array(list(itertools.permutations(range(5))))
def pair_f1(a,b):
    return 2*sum((Counter(a)&Counter(b)).values())/(len(a)+len(b))
def compare(pred,target):
    assert len(pred)==len(target)==5 and all(len(x)==8 for x in pred+target)
    matrix=np.array([[pair_f1(a,b) for b in target] for a in pred])
    i,j=linear_sum_assignment(-matrix);matched=float(matrix[i,j].mean())
    # A genuinely independent exhaustive search over all 120 assignments.
    brute=float(matrix[np.arange(5)[None,:],PERMS].mean(1).max())
    assert abs(matched-brute)<1e-12
    pooled=pair_f1([v for row in pred for v in row],[v for row in target for v in row])
    assert matched<=pooled+1e-12
    return pooled,matched
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    root=Path(root);out=root/'outputs/GenPlaylist-cue-matching-audit-20261112';out.mkdir(exist_ok=True)
    sources=json.loads(gzip.decompress((out/'source-plans.json.gz').read_bytes()))
    result={};arrays={};frequencies={}
    for ds in ['mpd','m4a']:
        ev=root/'work/sampled-candidates-20261110/complete-evidence';cand=json.loads((ev/ds/'candidates.json').read_text());ic=json.loads((ev/f'{ds}-item2cues.json').read_text())
        vp=root/'work/migration/original-ncl'/('src/02_creative_cues/outputs/production/20260804_172128/cue_vocab.json' if ds=='mpd' else 'data/processed/cues-music4all-onion-v3-u500/latest/cue_vocab.json');vocab=json.loads(vp.read_text())
        paths=[ev/ds/'candidates.json',ev/f'{ds}-item2cues.json',vp];report={'contexts':941,'catalog_size':len(cand['catalog_ids']),'methods':{},'provenance':{str(p.relative_to(root)):sha(p) for p in paths}}
        cases={k:[d] for k,d in sources[ds].items()}
        for method,key in [('uniform_random','random'),('train_popularity','popularity')]:
            cases[key]=[dict(source=str(p),sha256=sha(p),predictions=json.loads(p.read_text())['predictions']) for p in sorted((root/f'outputs/GenPlaylist-simple-baselines-20261112/{ds}/full_catalog/{method}').glob('seed*.json'))]
        if ds=='m4a':
            for arm in ['sid','warm','ramp','project']:
                p=root/f'outputs/GenPlaylist-tiger-preserve-m4a-20261112/evidence/full/{arm}/test-step2000.json';cases['trial_'+arm]=[dict(source=str(p),sha256=sha(p),predictions=json.loads(p.read_text())['predictions'])]
        for name,runs in cases.items():
            vals=[];provenance=[]
            for run in runs:
                d=run['predictions'];assert d['target_item_ids']==cand['truth'] and len(d['item_ids'])==941
                retrieved=name in ['random','popularity','ddbc_sid','tiger_sid_top5','trial_sid']
                predicted=[[ic[s][:8] for s in row] for row in d['item_ids']] if retrieved else d['cue_ids']
                target=[[ic[s][:8] for s in row] for row in cand['truth']]
                values=np.array([compare(p,t) for p,t in zip(predicted,target)]);vals.append(values);provenance.append({k:run[k] for k in ['source','sha256']})
            v=np.stack(vals);mean=v.mean(0);arrays[ds+'__'+name]=v
            report['methods'][name]={'pooled_f1':float(mean[:,0].mean()),'matched_f1':float(mean[:,1].mean()),'delta':float((mean[:,1]-mean[:,0]).mean()),'runs':len(runs),'seed_sd_pooled':float(v[:,:,0].mean(1).std(ddof=1)) if len(runs)>1 else None,'seed_sd_matched':float(v[:,:,1].mean(1).std(ddof=1)) if len(runs)>1 else None,'cue_source':'catalog cues of selected songs' if retrieved else 'model-predicted cues','sources':provenance}
            print(ds,name,round(mean[:,0].mean()*100,3),round(mean[:,1].mean()*100,3),flush=True)
        # Show the current population and selected fixed popularity songs without filtering words.
        pop=cases['popularity'][0]['predictions']['item_ids'][0];assert all(row==pop for row in cases['popularity'][0]['predictions']['item_ids'])
        freq=Counter(c for s in cand['catalog_ids'] for c in ic[s][:8]);targets=Counter(c for row in cand['truth'] for s in row for c in ic[s][:8]);pf=Counter(c for s in pop for c in ic[s][:8])
        frequencies[ds]={'catalog_size':len(cand['catalog_ids']),'catalog_cue_occurrences':sum(freq.values()),'unique_cues_observed':len(freq),'top10_occurrence_share':sum(n for _,n in freq.most_common(10))/sum(freq.values()),'popularity_songs':pop,'catalog':[dict(id=int(k),phrase=vocab[k],count=n,song_fraction=n/len(cand['catalog_ids'])) for k,n in freq.most_common()],'target_occurrences':{str(k):v for k,v in targets.items()},'popularity_occurrences':{str(k):v for k,v in pf.items()}}
        # Fraction of the old Popularity overlap accounted for by the top 10 catalog cues.
        top=set(k for k,n in freq.most_common(10));overlap=Counter()
        for row in cand['truth']:overlap.update(pf&Counter(c for s in row for c in ic[s][:8]))
        report['popularity_overlap_top10_fraction']=sum(n for k,n in overlap.items() if k in top)/sum(overlap.values())
        result[ds]=report
    protocol={'metric':'Item-matched Cue F1 (optimal assignment)','pair_score':'multiset F1 between eight predicted and eight target cues','playlist_score':'maximum mean pair score over one-to-one assignments between five predicted and five target slots','target_repeats':'preserved as separate target occurrences','cohort':'941 fixed contexts per dataset, full catalog only','predictions':'frozen, no retraining, no changes to selected songs or cues','random':'mean over original ten fixed seeds; no seed selection','verification':'Hungarian solution compared against exhaustive all120 permutations for EVERY context/run; pooled>=matched asserted','meaning':'target-cue agreement, not grounding against the output song, not generated-audio quality','wordcloud':'one count per selected catalog cue occurrence; first8 cues per catalog song, entire eligible catalog; full phrases retained, no manual exclusions; display top80 phrases','reporting':'diagnostic before manuscript adoption; originals retained; no paper edit/push/sync'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(out/'cue-frequencies.json').write_text(json.dumps(frequencies,indent=2,ensure_ascii=False)+'\n');np.savez_compressed(out/'per-context.npz',**arrays)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);main(p.parse_args().workspace)

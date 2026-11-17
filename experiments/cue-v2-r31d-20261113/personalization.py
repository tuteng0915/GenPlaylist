"""941-context, one-to-one cue matching: HistoryCue versus training-only popularity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import collections,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from datasets import load_from_disk
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-r31d-20261113'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import atomic,sha,paths

def f1(a,b):
    a=set(a)-{0};b=set(b)-{0};return 2*len(a&b)/(len(a)+len(b)) if a or b else 0.

def matched(a,b):
    matrix=np.array([[f1(x,y) for y in b] for x in a]);r,c=linear_sum_assignment(-matrix);return float(matrix[r,c].sum()/5)

def main():
    started=time.time();report={}
    for ds in ['mpd','m4a']:
        x=paths(ds);co=json.loads((R/'outputs/sampled-candidates-20261110'/ds/'candidates.json').read_text());assert len(co['histories'])==941
        split_path=R/'experiments'/('combined-decoding-20261107/split.json' if ds=='mpd' else 'two-dataset-fair-20261108/split.json');split=json.loads(split_path.read_text());prepared=x['prepared'];assert sha(prepared/'prepared_manifest.json')==split['source_manifest_sha256']
        raw=load_from_disk(str(prepared/'raw_dataset'))['train'];counts=collections.Counter(str(i) for row in raw.select(split['training_indices'])['item_seq'] for i in row);catalog=set(co['catalog_ids']);popular=[i for i,_ in sorted(counts.items(),key=lambda p:(-p[1],p[0])) if i in catalog][:5];assert len(popular)==5
        dsout=dict(contexts=941,training_split_sha256=sha(split_path),popular_songs=popular,definition='HistoryCue: most frequent8 cues among15 history songs, deterministic ID ties, repeated as five cue sets for matched F1. Popularity: top5 training-frequency songs, no test frequencies; use their catalog cues. Additional GlobalCueFrequency is labeled separately.',arms={})
        for version,p in [('v1',x['cues']/'item2cues.json'),('r31',O/'v2'/ds/'item2cues.json')]:
            m={i:row[:8] for i,row in json.loads(p.read_text()).items()};globalcount=collections.Counter()
            for i,n in counts.items():
                if i in m:
                    for c in m[i]:
                        if c:globalcount[c]+=n
            global8=[c for c,_ in sorted(globalcount.items(),key=lambda p:(-p[1],p[0]))[:8]];hist=[];pop=[];globalpop=[]
            for h,t in zip(co['histories'],co['truth']):
                cnt=collections.Counter(c for i in h for c in m[str(i)] if c);cue=[c for c,_ in sorted(cnt.items(),key=lambda p:(-p[1],p[0]))[:8]];target=[m[str(i)] for i in t]
                hist.append(matched([cue]*5,target));pop.append(matched([m[i] for i in popular],target));globalpop.append(matched([global8]*5,target))
            delta=np.asarray(hist)-pop;rng=np.random.default_rng(42);boot=np.concatenate([delta[rng.integers(len(delta),size=(100,len(delta)))].mean(1) for _ in range(100)])
            dsout['arms'][version]=dict(history_cue_f1=float(np.mean(hist)),popularity_song_cue_f1=float(np.mean(pop)),global_cue_frequency_f1=float(np.mean(globalpop)),history_minus_popularity_song=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),map_sha256=sha(p));atomic(O/'v2'/ds/(version+'-personalization-rows.json'),dict(history=hist,popularity_song=pop,global_cue_frequency=globalpop))
        report[ds]=dsout
    atomic(O/'personalization.json',dict(datasets=report,elapsed_seconds=time.time()-started,training_started=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'personalization-failure.json',dict(traceback=traceback.format_exc()));raise

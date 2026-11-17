"""CPU-only Stage A content ruler on the frozen 11292 first-item records."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys, json, time, collections
from pathlib import Path
import numpy as np
R = Path('/storage_fast/ttu/GenPlaylist')
S = R/'outputs/cue-v2-stageA-20261112'
O = R/'outputs/first-item-independent-content-20261113'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import module, paths, lyric_path, lyric_quality, sha, atomic, immutable
from metrics import text_embedding
lc=module('first_item_list5_paths',R/'experiments/list5-audio-20261111/common.py')
ARMS={'predicted':'ddbc_joint','none':'drop','catalog':'retrieved','ddbc_sid_only':'ddbc_sid','tiger_sid_only':'tiger_sid','history_only':'direct'}

def root(ds,arm):
    if arm=='predicted':return R/('outputs/generation-final-20261107/full' if ds=='mpd' else 'outputs/two-dataset-fair-20261108/m4a/audio/full')
    return lc.old_root(ds,ARMS[arm])

def embeddings(model,p,texts,provenance):
    immutable(p.with_suffix('.json'),provenance)
    if p.exists():return np.load(p)
    a=text_embedding(model,texts)
    np.save(p,a)
    return a

def main():
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    started=time.time();O.mkdir(exist_ok=True)
    model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device='cpu',local_files_only=True)
    immutable(O/'config.json',dict(scope='11292 frozen first-item records; six arms x941 x2datasets',encoder='sentence-transformers/all-mpnet-base-v2',device='cpu',threads=2,embedding_implementation_sha256=sha(R/'experiments/cue-v2-stageA-20261112/metrics.py'),reference_protocol_sha256=sha(R/'experiments/cue-v2-followup-20261113/independent_content.py'),arm_mapping=ARMS,bootstrap_samples=10000,seed=42,mask='fixed by actual reference availability across all arms; average available targets out of five',caveat='Generated lyric_draft/music_attributes are renderer text, not transcribed/inferred audio content. Legacy first-item model checkpoints are retained; not new v2 results.'))
    for ds in ['mpd','m4a']:
        out=O/ds;out.mkdir(exist_ok=True)
        cp=R/'outputs/sampled-candidates-20261110'/ds/'candidates.json';co=json.loads(cp.read_text())
        assert len(co['truth'])==len(co['histories'])==941
        gate=json.loads((S/ds/'cohort.json').read_text())
        for row in gate['gate_rows']:
            i=row['source_history_index'];assert row['targets']==co['truth'][i] and row['history']==co['histories'][i]
        meta=json.loads((paths(ds)['data']/'catalog_metadata.json').read_text())
        ids=sorted({str(i) for row in co['truth'] for i in row});lookup={i:j for j,i in enumerate(ids)}
        lyrics=[];attrs=[];lv=[];mv=[];qualities=collections.Counter();refs={}
        for i in ids:
            m=meta[i];p=lyric_path(ds,i,m);raw=p.read_text(errors='replace') if p else '';q=lyric_quality(raw);qualities[q]+=1
            text='\n'.join(dict.fromkeys(l.strip() for l in raw.splitlines() if l.strip())) if q not in ['missing_or_short','suspected_listing'] else ''
            att='; '.join(str(v) for v in [m.get('genre',''),', '.join(m.get('tags',[])),m.get('mood','')] if v)
            lyrics.append(text);attrs.append(att);lv.append(bool(text));mv.append(bool(att));refs[i]=dict(lyric_sha256=sha(p) if p else None,quality=q)
        reference_provenance=dict(ids=ids,refs=refs,metadata_sha256=sha(paths(ds)['data']/'catalog_metadata.json'),cohort_sha256=sha(cp))
        le=embeddings(model,out/'reference-lyrics.npy',lyrics,reference_provenance)
        me=embeddings(model,out/'reference-metadata.npy',attrs,reference_provenance)
        ix=np.asarray([[lookup[str(i)] for i in row] for row in co['truth']]);masks={'actual_lyric_fit':np.asarray(lv)[ix],'actual_metadata_fit':np.asarray(mv)[ix]}
        allout={};report=dict(dataset=ds,contexts=941,source_config_sha256=sha(O/'config.json'),reference_qualities=dict(qualities),arms={},paired={})
        for arm in ARMS:
            base=root(ds,arm);files=sorted((base/'verbalization').glob('*/*.json'),key=lambda p:int(p.stem));assert len(files)==941,(ds,arm,len(files));assert [int(p.stem) for p in files]==list(range(941))
            rows=[json.loads(p.read_text()) for p in files]
            for i,row in enumerate(rows):
                assert row['example_index']==i and list(map(str,row['reference_item_ids']))==list(map(str,co['histories'][i])),(ds,arm,i,'history')
                assert str(row['target_item_id'])==str(co['truth'][i][0]),(ds,arm,i,'target')
            prov=dict(root=str(base),files={str(p.relative_to(base)):sha(p) for p in files})
            ge=embeddings(model,out/(arm+'-lyrics.npy'),[r['lyric_draft'] for r in rows],prov)
            ga=embeddings(model,out/(arm+'-attributes.npy'),[r['music_attributes'] for r in rows],prov)
            report['arms'][arm]={'source':str(base),'records':941,'source_manifest_sha256':sha((out/(arm+'-lyrics.npy')).with_suffix('.json'))}
            for name,generated,reference in [('actual_lyric_fit',ge,le),('actual_metadata_fit',ga,me)]:
                sim=np.einsum('nd,nkd->nk',generated,reference[ix]);mask=masks[name];ok=mask.sum(1)>0;values=(sim*mask).sum(1)/np.maximum(mask.sum(1),1)
                allout[arm+'__'+name]=values;allout[name+'__valid']=ok;allout[arm+'__'+name+'_all5_missing_zero']=(sim*mask).mean(1)
                report['arms'][arm][name]=dict(mean=float(values[ok].mean()),contexts=int(ok.sum()),reference_occurrences=int(mask.sum()))
            atomic(out/'progress.json',dict(completed_arm=arm,elapsed_seconds=time.time()-started))
            print('CONTENT_ARM',ds,arm,time.time()-started,flush=True)
        for other in [a for a in ARMS if a!='predicted']:
            pair=report['paired']['predicted minus '+other]={}
            for name in masks:
                delta=(allout['predicted__'+name]-allout[other+'__'+name])[allout[name+'__valid']];rng=np.random.default_rng(42);boot=[]
                for _ in range(50):boot.extend(delta[rng.integers(len(delta),size=(200,len(delta)))].mean(1))
                pair[name]=dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),contexts=len(delta))
        np.savez_compressed(out/'per-context.npz',**allout);report['elapsed_seconds']=time.time()-started;atomic(out/'report.json',report)
    atomic(O/'complete.json',dict(records=11292,elapsed_seconds=time.time()-started,stage_B_started=False,paper_changed=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'failure.json',dict(traceback=traceback.format_exc()));raise

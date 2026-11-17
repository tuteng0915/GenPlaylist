"""CPU gate-0 selection, loss audit and author inspection table. Never starts full extraction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,time,collections,html,types,datetime
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';O=R/'outputs/cue-v2-r3-author-pilot-20261113';E=R/'experiments/cue-v2-stageA-20261112'
sys.path.insert(0,str(E));from common import atomic,sha,OFFENSIVE,paths
from build import choose,unit

def reason_groups(counter):
    groups=collections.Counter()
    for why,n in counter.items():
        name=('names' if why in ['artist_name','catalog_person_name'] else 'song_title' if why in ['title_copy','song_title'] else 'offensive' if 'offensive' in why else 'length' if 'length' in why else 'verbatim_lyrics' if 'verbatim' in why else 'evidence_check' if why in ['unsupported_quote','evidence_id','evidence_quote','unusable_lyrics','sound_without_metadata','theme_without_lyrics','unstated_acoustic_feature','uninformative_mood','invalid_evidence_id'] else why)
        groups[name]+=n
    return dict(groups)

def main():
    started=time.time();deadline=datetime.datetime.fromisoformat('2026-11-13T11:11:30+08:00').timestamp()
    while not all((O/d/'extraction-complete.json').exists() for d in ['mpd','m4a']):
        if any((O/d/'failure.json').exists() for d in ['mpd','m4a']):raise RuntimeError('Pilot extraction failed')
        if time.time()>=deadline:raise TimeoutError('Pilot deadline; no gate advancement')
        time.sleep(10)
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);model.max_seq_length=256
    records={};keys=set();cohorts={};inputs={};loss={};report={};mapout={};selected={}
    for ds in ['mpd','m4a']:
        co=cohorts[ds]=json.loads((S/ds/'cohort.json').read_text());inputs[ds]=json.loads((S/ds/'inputs.json').read_text())
        original=collections.Counter();oldpilot=collections.Counter();newtheme=collections.Counter();newmeta=collections.Counter()
        for i in co['extraction_ids']:
            d=json.loads((S/ds/'candidates'/f'{i}.json').read_text());original.update(c['reason'] for c in d['rejected'])
            if i in co['pilot']:oldpilot.update(c['reason'] for c in d['rejected'])
        for i in co['pilot']:
            p=O/ds/'candidates'/f'{i}.json';d=records[(ds,i)]=json.loads(p.read_text());assert d['input']==inputs[ds][i];assert d['config_sha256']==sha(O/ds/'config.json')
            for c in d['accepted']:keys.add((c['category'],c['phrase']))
            for c in d['rejected']:(newtheme if c.get('category')=='theme' else newmeta)[c['reason']]+=1
        loss[ds]=dict(original_scope_songs=len(co['extraction_ids']),original_scope_reasons=dict(original),original_scope_grouped=reason_groups(original),original_same100_reasons=dict(oldpilot),new_theme_reasons=dict(newtheme),new_metadata_reasons=dict(newmeta),new_theme_grouped=reason_groups(newtheme),counting='One primary recorded rejection reason per candidate. Metadata input tags and LLM theme candidates have different denominators; do not compare raw drop totals as rates.')
    keys=sorted(keys);pos={key:i+1 for i,key in enumerate(keys)};vectors=np.vstack([np.zeros(384),unit(model.encode([k[1] for k in keys],batch_size=128,convert_to_numpy=True))]);occ=[pos[(c['category'],c['phrase'])] for d in records.values() for c in d['accepted']];mean=vectors[occ].mean(0)
    for ds in ['mpd','m4a']:
        ids=cohorts[ds]['pilot'];selected[ds]={};mapout[ds]={};counts=collections.Counter();cats=collections.Counter();groups={}
        for i in ids:
            d=records[(ds,i)];mapped={}
            for c in d['accepted']:
                j=pos[(c['category'],c['phrase'])];r=dict(c,cue_id=j,canonical_phrase=c['phrase'],mapping_score=(1-float(vectors[j]@mean))*c['confidence'])
                if j not in mapped or mapped[j]['mapping_score']<r['mapping_score']:mapped[j]=r
            choices=choose(list(mapped.values()),vectors);selected[ds][i]=choices;mapout[ds][i]=[c['phrase'] for c in choices]+['<unk>']*(8-len(choices));cats.update(c['category'] for c in choices)
            counts['candidate_count']+=len(d['accepted']);counts['selected_slots']+=len(choices);counts['mapped_unique_candidates']+=len(mapped)
            if d['input']['lyrics']:
                n=sum(c['category']=='theme' for c in d['accepted']);counts['lyric_songs']+=1;counts['lyric_songs_theme_lt4']+=n<4;counts['lyric_songs_theme_ge6']+=n>=6
            counts['songs_without_emotion_metadata']+=not any(c['category']=='emotion' for c in d['accepted'])
        for label,subset in [('all',ids),('with_lyrics',[i for i in ids if inputs[ds][i]['lyrics']]),('without_lyrics',[i for i in ids if not inputs[ds][i]['lyrics']])]:
            ns=[len(selected[ds][i]) for i in subset];groups[label]=dict(songs=len(ns),nonempty_slots=sum(ns),possible_slots=8*len(ns),coverage=sum(ns)/max(1,8*len(ns)),min_slots=min(ns) if ns else None,songs_below4=sum(n<4 for n in ns),slot_histogram=dict(collections.Counter(ns)))
        old=json.loads((S/ds/'extraction-config.json').read_text()) if (S/ds/'extraction-config.json').exists() else {}
        report[ds]=dict(counts=counts,selected_categories=cats,coverage=groups,gate1_threshold_preview=dict(lyrics85=groups['with_lyrics']['coverage']>=.85,no_lyrics_each4=groups['without_lyrics']['songs_below4']==0),extraction_runtime=json.loads((O/ds/'extraction-complete.json').read_text()))
        atomic(O/ds/'selected-evidence.json',selected[ds]);atomic(O/ds/'selected-cues.json',mapout[ds])
    atomic(O/'filter-loss.json',loss)
    atomic(O/'coverage.json',dict(datasets=report,scope='Gate0 pilot; selection on extracted free phrases before full-catalog shared-vocabulary clustering. Not gate1 mapped coverage.',selection='MMR lambda.7;2/2/4 soft quotas; evidence-supported candidates only; no rare-cue boost',elapsed_seconds=time.time()-started,author_confirmation_required=True,full_catalog_started=False))
    # Reuse the identical original diagnostic definition; run on CPU and add the new pilot arm.
    source=(E/'diagnostics.py').read_text();m=types.ModuleType('author_pilot_stats');exec(compile(source.replace("device='cuda'","device='cpu'"),str(E/'diagnostics.py'),'exec'),m.__dict__);oldarms=m.arms
    def arms(ds):
        co,data=oldarms(ds);data['r3_metadata_theme']=mapout[ds];return co,data
    def save_diagnostic(p,x):
        assert Path(p).name=='diagnostics.json';x['source_sha256']=sha(E/'diagnostics.py');x['scope']='200song pilot, not full catalog';atomic(O/'diagnostics.json',x)
    m.arms=arms;m.atomic=save_diagnostic;m.stats()
    lines=['<!doctype html><meta charset="utf-8"><title>门槛0：40首cue对照</title><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px;vertical-align:top}td{max-width:400px}pre{white-space:pre-wrap}</style><h1>门槛0：v1 / MMR / r3</h1><p>每个数据集固定等间距20首，未按效果挑选。r3尚未经过全目录规范词聚类；冒犯词展示已遮蔽。真实性由作者检查，不将引文存在视为语义支持已证实。</p>'];audit=[]
    esc=lambda s:html.escape(OFFENSIVE.sub('[redacted]',str(s)))
    for ds in ['mpd','m4a']:
        co,data=oldarms(ds);chosen=[co['pilot'][int(i)] for i in np.linspace(0,99,20,dtype=int)];lines.append('<h2>'+ds+'</h2><table><tr><th>歌曲</th><th>v1</th><th>MMR</th><th>r3</th><th>r3证据 / 元数据</th></tr>')
        for i in chosen:
            inp=inputs[ds][i];ev=selected[ds][i];audit.append(dict(dataset=ds,item_id=i,input=inp,v1=data['v1'][i],mmr=data['mmr'][i],r3=mapout[ds][i],r3_evidence=ev))
            cells=[esc(i+' / '+inp['title'])]+['<br>'.join(esc(c) for c in row) for row in [data['v1'][i],data['mmr'][i],mapout[ds][i]]]+['<br>'.join(esc(c['category']+': '+c['phrase']+' ← '+c['source']+': '+c['evidence']) for c in ev)+'<details><summary>来源</summary><pre>'+esc(json.dumps(inp,ensure_ascii=False,indent=2))+'</pre></details>'];lines.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
        lines.append('</table>')
    (O/'author-review-40.html').write_text('\n'.join(lines));atomic(O/'author-review-40.json',audit)
    atomic(O/'gate0-ready.json',dict(songs=200,review_songs=40,elapsed_seconds=time.time()-started,author_confirmation_required=True,full_catalog_started=False,training_started=False,audio_started=False,evidence_sha256={p.name:sha(p) for p in [O/'coverage.json',O/'filter-loss.json',O/'diagnostics.json',O/'author-review-40.html']}))
    print('GATE0_READY_STOP_FOR_AUTHOR',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        atomic(O/'finish-failure.json',dict(traceback=traceback.format_exc()));raise

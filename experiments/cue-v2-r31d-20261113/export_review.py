"""Readable200pilot selection and fixed40song review; not post-vocabulary gate1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import json,sys,html,collections
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-r31d-20261113';S=R/'outputs/cue-v2-stageA-20261112'
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import atomic,paths,OFFENSIVE
from build import choose,unit
from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(2)
records={};cohorts={}
for ds in ['mpd','m4a']:
    co=cohorts[ds]=json.loads((S/ds/'cohort.json').read_text())
    for i in co['pilot']:records[ds,i]=json.loads((O/'pilot'/ds/'candidates'/f'{i}.json').read_text())
keys=sorted({(c['category'],c['phrase']) for d in records.values() for c in d['accepted']});pos={k:i+1 for i,k in enumerate(keys)}
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);model.max_seq_length=256;vec=np.vstack([np.zeros(384),unit(model.encode([p for c,p in keys],batch_size=128,convert_to_numpy=True))]);mean=vec[[pos[c['category'],c['phrase']] for d in records.values() for c in d['accepted']]].mean(0)
audit=[];coverage={};esc=lambda s:html.escape(OFFENSIVE.sub('[redacted]',str(s)))
lines=['<!doctype html><meta charset="utf-8"><title>r3.1固定40首复核</title><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px;vertical-align:top}td{max-width:400px}pre{white-space:pre-wrap}</style><h1>v1 / MMR / r3.1</h1><p>固定每数据集20首，不按效果挑选。下列r3.1为候选MMR选择，尚非规范词表映射结果。引用存在不等于语义真实性验证。展示中的冒犯词已遮蔽。</p>']
for ds in ['mpd','m4a']:
    ids=cohorts[ds]['pilot'];oldv=json.loads((paths(ds)['cues']/'cue_vocab.json').read_text());v1=json.loads((paths(ds)['cues']/'item2cues.json').read_text());mmr=json.loads((S/ds/'v1-mmr-item2cues.json').read_text());chosen={ids[int(j)] for j in np.linspace(0,99,20,dtype=int)};selection={}
    lines.append('<h2>'+ds+'</h2><table><tr><th>歌曲</th><th>v1</th><th>MMR</th><th>r3.1</th><th>证据</th></tr>')
    for i in ids:
        d=records[ds,i];cand=[dict(c,cue_id=pos[c['category'],c['phrase']],mapping_score=(1-float(vec[pos[c['category'],c['phrase']]]@mean))*c['confidence']) for c in d['accepted']];selected=choose(cand,vec);selection[i]=selected
        if i not in chosen:continue
        row=dict(dataset=ds,item_id=i,input=d['input'],v1=[oldv[c] for c in v1[i][:8]],mmr=[oldv[c] for c in mmr[i][:8]],r31=[c['phrase'] for c in selected]+['<unk>']*(8-len(selected)),evidence=selected,rewrites=d['rewrites']);audit.append(row)
        cells=[esc(i+' '+d['input']['title'])]+['<br>'.join(esc(c) for c in row[a]) for a in ['v1','mmr','r31']]+['<br>'.join(esc(c['phrase']+' ← '+c['evidence']) for c in selected)];lines.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    lines.append('</table>');atomic(O/'pilot'/ds/'selected-evidence.json',selection)
    coverage[ds]={}
    for label,sub in [('with_lyrics',[i for i in ids if records[ds,i]['input']['lyrics']]),('without_lyrics',[i for i in ids if not records[ds,i]['input']['lyrics']])]:
        coverage[ds][label]=dict(songs=len(sub),selected_slots=sum(len(selection[i]) for i in sub),possible_slots=len(sub)*8,coverage=sum(len(selection[i]) for i in sub)/max(1,len(sub)*8))
atomic(O/'review-40.json',audit);(O/'review-40.html').write_text('\n'.join(lines));atomic(O/'pilot-selection-coverage.json',dict(scope='precanonical only',datasets=coverage))
print('REVIEW_READY',len(audit),flush=True)

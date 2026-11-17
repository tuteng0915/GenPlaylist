import itertools, json
from collections import Counter
from pathlib import Path
import numpy as np

R=Path('/storage_fast/ttu/GenPlaylist')
CATS=json.loads((R/'outputs/cue-v2-r31e-20261113/cue_categories.json').read_text())
PERMS=list(itertools.permutations(range(5)))

def f1(pred, truth, cat=None):
    if cat: pred=[x for x in pred if CATS[x]==cat]; truth=[x for x in truth if CATS[x]==cat]
    a,b=Counter(pred),Counter(truth); hit=sum((a&b).values()); den=len(pred)+len(truth)
    return 0. if not den else 2*hit/den

for ds in ['mpd','m4a']:
    p=R/'outputs/cue-v2-r31e-tiger-20261113'/ds/'joint/audio/plan.json'
    q=json.loads(p.read_text())['predictions']; item2c=json.loads((R/'outputs/cue-v2-r31e-20261113'/ds/'item2cues.json').read_text())
    rows=[]
    for pred,targets in zip(q['cue_ids'],q['target_item_ids']):
        truth=[item2c[str(x)][:8] for x in targets]
        scores=np.array([[f1(pred[i],truth[j]) for j in range(5)] for i in range(5)])
        perm=max(PERMS,key=lambda z:sum(scores[i,j] for i,j in enumerate(z)))
        row={'overall':float(np.mean([scores[i,j] for i,j in enumerate(perm)]) )}
        for c in ['sound','emotion','theme']: row[c]=float(np.mean([f1(pred[i],truth[j],c) for i,j in enumerate(perm)]))
        rows.append(row)
    out={'dataset':ds,'model':'TIGER+cues r3.1e','contexts':len(rows),'matching':'maximum-weight one-to-one matching of five predicted songs to five target songs; edge score is within-song cue multiset F1','means':{k:float(np.mean([x[k] for x in rows])) for k in rows[0]},'per_context':rows}
    dest=R/'outputs/cue-v2-r31e-evaluation-20261115'/ds/'tiger-joint-one-to-one.json'; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n');print(ds,out['means'])

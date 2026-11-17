import itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');cats=json.loads((R/'outputs/cue-v2-r31e-20261113/cue_categories.json').read_text());perms=list(itertools.permutations(range(5)))
def f(a,b,c=None):
 if c:a=[x for x in a if cats[x]==c];b=[x for x in b if cats[x]==c]
 x,y=Counter(a),Counter(b);return 0 if len(a)+len(b)==0 else 2*sum((x&y).values())/(len(a)+len(b))
for ds in ['mpd','m4a']:
 q=json.loads((R/'outputs/cue-v2-r31e-evaluation-20261115'/ds/'ddbc-joint-941.json').read_text())['predictions'];m=json.loads((R/'outputs/cue-v2-r31e-20261113'/ds/'item2cues.json').read_text()); rows=[]
 for a,t in zip(q['cue_ids'],q['target_item_ids']):
  b=[m[str(x)][:8] for x in t];s=[[f(a[i],b[j]) for j in range(5)] for i in range(5)];z=max(perms,key=lambda p:sum(s[i][j] for i,j in enumerate(p)));rows.append({k:float(np.mean([f(a[i],b[j],None if k=='overall' else k) for i,j in enumerate(z)])) for k in ['overall','sound','emotion','theme']})
 out={'dataset':ds,'model':'DDBC+cues r3.1e','contexts':941,'matching':'maximum-weight one-to-one matching by within-song cue multiset F1','means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},'per_context':rows};p=R/'outputs/cue-v2-r31e-evaluation-20261115'/ds/'ddbc-joint-one-to-one.json';p.write_text(json.dumps(out,indent=2)+'\n');print(ds,out['means'])

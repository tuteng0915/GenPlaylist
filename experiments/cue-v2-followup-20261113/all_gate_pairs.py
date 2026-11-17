"""All six arm contrasts, including both retained cue-reference Content fit scores."""
from common_followup import *
import itertools
for ds in ['mpd','m4a']:
 r=json.loads((S/ds/'gate-report.json').read_text());z=np.load(S/ds/'gate-per-context.npz');out={'dataset':ds,'contexts':200,'source_sha256':sha(S/ds/'gate-per-context.npz'),'pairs':{}}
 metrics=[k.split('__',1)[1] for k in z.files if k.startswith('v1__') and not k.endswith('fad_bootstrap')]
 for left,right in itertools.combinations(ARMS,2):
  scores={}
  for metric in metrics:
   d=z[left+'__'+metric]-z[right+'__'+metric];assert d.shape==(200,)
   scores[metric]={'delta':float(d.mean()),'ci95':interval(d)}
  scores['fad']={'delta':r['arms'][left]['fad']-r['arms'][right]['fad'],'ci95':np.quantile(z[left+'__fad_bootstrap']-z[right+'__fad_bootstrap'],[.025,.975]).tolist()};out['pairs'][left+' minus '+right]=scores
 save(ds+'/all-gate-pairs.json',out)
print('ALL_GATE_PAIRS_DONE',flush=True)

"""Reuse paired bootstrap draws; no audio regeneration or new random sampling."""
from common_followup import *
for ds in ['mpd','m4a']:
 waiting([S/ds/'gate-report.json',S/ds/'gate-per-context.npz'])
 r=json.loads((S/ds/'gate-report.json').read_text());a=np.load(S/ds/'gate-per-context.npz');out={'dataset':ds,'contexts':200,'pairs':{},'source_report_sha256':sha(S/ds/'gate-report.json'),'definition':'first true target conditioned gate; paired context bootstrap,10000 scalar/1000 FAD; positive FAD delta is worse'}
 for left,right in [('v1','none'),('mmr','none'),('mmr','v1'),('v2','none'),('v2','v1')]:
  p={}
  for metric in ['mert_oas','history_fit']:
   d=a[left+'__'+metric]-a[right+'__'+metric];p[metric]={'delta':float(d.mean()),'ci95':interval(d)}
  d=a[left+'__fad_bootstrap']-a[right+'__fad_bootstrap'];p['fad']={'delta':r['arms'][left]['fad']-r['arms'][right]['fad'],'ci95':np.quantile(d,[.025,.975]).tolist()};out['pairs'][left+' minus '+right]=p
 save(ds+'/paired-gate.json',out)
print('PAIRS_DONE',flush=True)

"""Complete all pair contrasts and independently check reported means/deltas."""
from common_followup import *
import itertools
for ds in ['mpd','m4a']:
 waiting([O/ds/'independent-content-report.json',O/ds/'independent-content-per-context.npz'])
 q=np.load(O/ds/'independent-content-per-context.npz');report=json.loads((O/ds/'independent-content-report.json').read_text());allpairs={}
 for l,r in itertools.combinations(ARMS,2):
  pair={}
  for k in ['actual_lyric_fit','actual_metadata_fit']:
   valid=q[k+'__valid'];x=q[l+'__'+k][valid];y=q[r+'__'+k][valid];d=x-y;assert abs(x.mean()-report['arms'][l][k]['mean'])<1e-10
   rng=np.random.default_rng(42);b=[]
   for _ in range(50):b.extend(d[rng.integers(len(d),size=(200,len(d)))].mean(1))
   pair[k]={'delta':float(d.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'contexts':len(d)}
  allpairs[l+' minus '+r]=pair
 report['all_six_paired_contrasts']=allpairs;report['independent_mean_check_passed']=True;report['source_vectors_sha256']=sha(O/ds/'independent-content-per-context.npz');report['encoder_assets']=json.loads((S/'text-encoder-assets.json').read_text());save(ds+'/independent-content-complete.json',report)
print('ALL_CONTENT_PAIRS_VERIFIED',flush=True)

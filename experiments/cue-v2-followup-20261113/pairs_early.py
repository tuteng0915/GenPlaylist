"""Immediate v1-vs-none gate, without waiting for MMR. Exact frame sufficient statistics."""
from common_followup import *
from metrics import fad
from scipy.optimize import linear_sum_assignment

def summaries(frames):
 return np.array([len(x) for x in frames]),np.stack([x.sum(0) for x in frames]),np.stack([x.T@x for x in frames])
def gaussian(s,weights):
 n,sums,xx=s;count=float(weights@n);mu=weights@sums/count;cov=(np.einsum('n,nij->ij',weights,xx)-count*np.outer(mu,mu))/(count-1);return mu,(cov+cov.T)/2
def from_gaussians(a,b):
 mx,cx=a;my,cy=b;val,vec=np.linalg.eigh(cx);rt=(vec*np.sqrt(np.maximum(val,0)))@vec.T;pr=rt@cy@rt;return float(max(0,np.sum((mx-my)**2)+np.trace(cx)+np.trace(cy)-2*np.sqrt(np.maximum(np.linalg.eigvalsh((pr+pr.T)/2),0)).sum()))
for ds in ['mpd','m4a']:
 required=[S/ds/'gate'/a/'metrics/mert-complete.json' for a in ['v1','none']]+[S/ds/'vggish'/a/'complete.json' for a in ['v1','none','reference']];waiting(required)
 co=json.loads((S/ds/'cohort.json').read_text());x=paths(ds);cat=x['catalog'];ids=json.loads((cat/'catalog_item_ids.json').read_text());lk={str(i):j for j,i in enumerate(ids)};e=unit(np.load(cat/'catalog_mert_embeddings_l2.npy'));per={};vg={a:[np.load(S/ds/'vggish'/a/f'{i:04d}.npz')['embedding'].astype(float) for i in range(200)] for a in ['v1','none','reference']};summ={a:summaries(v) for a,v in vg.items()}
 for a in ['v1','none']:
  b=S/ds/'gate'/a;pr=unit(np.stack([np.load(b/'metrics/mert-rows'/f'{i:04d}.npz')['embedding'] for i in range(200)]));oas=[];hist=[]
  for v,r in zip(pr,co['gate_rows']):
   sim=v[None]@e[[lk[i] for i in r['targets']]].T;row,col=linear_sum_assignment(-sim);oas.append(float(sim[row,col].mean()));hist.append(float((e[[lk[i] for i in r['history']]]@v).mean()))
  per[a]={'mert_oas':np.array(oas),'history_fit':np.array(hist)}
 weights=np.ones(200);meanfad={a:from_gaussians(gaussian(summ[a],weights),gaussian(summ['reference'],weights)) for a in ['v1','none']}
 for a in ['v1','none']:assert abs(meanfad[a]-fad((np.concatenate(vg[a]),np.concatenate(vg['reference']))))<1e-7
 rng=np.random.default_rng(42);boot={a:[] for a in ['v1','none']}
 for j in range(1000):
  ix=rng.integers(200,size=200);w=np.bincount(ix,minlength=200);ref=gaussian(summ['reference'],w)
  for a in ['v1','none']:
   v=from_gaussians(gaussian(summ[a],w),ref);boot[a].append(v)
   if j==0:assert abs(v-fad((np.concatenate([vg[a][i] for i in ix]),np.concatenate([vg['reference'][i] for i in ix]))))<1e-7
 out={'dataset':ds,'contexts':200,'contrast':'v1 minus no cues','means':{a:{**{k:float(v.mean()) for k,v in per[a].items()},'fad':meanfad[a]} for a in ['v1','none']},'paired':{},'bootstrap':{'unit':'context','scalar':10000,'fad':1000,'seed':42,'same_draws_for_both_arms_and_reference':True},'validation':'Hungarian OAS recomputed; FAD sufficient-statistics agrees with direct concatenation for full sample and first bootstrap draw'}
 for k in per['v1']:
  d=per['v1'][k]-per['none'][k];out['paired'][k]={'delta':float(d.mean()),'ci95':interval(d)}
 out['paired']['fad']={'delta':meanfad['v1']-meanfad['none'],'ci95':np.quantile(np.array(boot['v1'])-boot['none'],[.025,.975]).tolist()};save(ds+'/v1-vs-none-immediate.json',out);print('IMMEDIATE_PAIRS_DONE',ds,flush=True)

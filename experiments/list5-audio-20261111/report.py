import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();x=paths(a.dataset);co=json.loads((O/a.dataset/'cohort.json').read_text());rows=co['rows'];cat=x['catalog'];manifest=json.loads((cat/'mert_manifest.json').read_text());idpath=cat/'catalog_item_ids.json';vecpath=cat/'catalog_mert_embeddings_l2.npy'
assert sha(idpath)==manifest['outputs'][idpath.name] and sha(vecpath)==manifest['outputs'][vecpath.name]
ids=json.loads(idpath.read_text());lookup={str(v):i for i,v in enumerate(ids)};emb=np.load(vecpath);refs=np.array([[lookup[s] for s in r['history']] for r in rows]);out=dict(dataset=a.dataset,lists=100,items_per_list=5,clips_per_arm=500,cohort_sha256=sha(O/a.dataset/'cohort.json'),ci_unit='list (five item scores averaged first)',arms={},paired={});per={}
def ci(x):
 rng=np.random.default_rng(42);assert len(x)==100
 return np.quantile(x[rng.integers(100,size=(10000,100))].mean(1),[.025,.975]).tolist()
for arm in ARMS:
 b=O/a.dataset/arm;system=SYSTEMS[arm];mert=np.stack([np.load(b/'metrics/mert-rows'/f'{i:04d}.npz')['embedding'] for i in range(500)]);history=np.einsum('bd,bkd->bk',mert,emb[refs]).mean(1).reshape(100,5).mean(1)
 slug=system.lower().replace('-','_');audio=np.load(b/'metrics/clap'/f'{slug}_clap_audio_l2.npy');text=np.load(b/'metrics/clap'/f'{slug}_clap_attribute_l2.npy');clap=np.einsum('bd,bd->b',audio,text).reshape(100,5).mean(1)
 fad=json.loads((b/'metrics/fad.json').read_text())['fad'][system];per[arm]=dict(history_fit=history,clap_adherence=clap);out['arms'][arm]=dict(means={k:float(v.mean()) for k,v in per[arm].items()},ci95={k:ci(v) for k,v in per[arm].items()},fad=fad)
for model in ['ddbc','tiger']:
 for comparator in [model+'_sid','direct','history_random','profile']:
  key=model+'_joint minus '+comparator;out['paired'][key]={k:dict(difference=float((per[model+'_joint'][k]-per[comparator][k]).mean()),ci95=ci(per[model+'_joint'][k]-per[comparator][k])) for k in ['history_fit','clap_adherence']}
np.savez(O/a.dataset/'per-list.npz',**{arm+'__'+k:v for arm,d in per.items() for k,v in d.items()});atomic(O/a.dataset/'report.json',out);print(json.dumps(out,indent=2))

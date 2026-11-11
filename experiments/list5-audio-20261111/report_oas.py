"""Five generated audios vs five target songs; global FAD remains unassigned."""
import argparse
from common import *
from oas import assignment_oas, summary, normalized


def main(ds):
    x=paths(ds); cp=O/ds/'cohort.json'; co=json.loads(cp.read_text()); rows=co['rows']
    assert len(rows)==500
    assert sha(x['art']/'item_id_to_row.json')==json.loads((x['catalog']/'mert_manifest.json').read_text())['catalog_mapping_sha256']
    for i, r in enumerate(rows):
        assert r['flat_index']==i and r['list_index']==i//5 and r['item_position']==i%5
    cat=x['catalog']; mp=cat/'mert_manifest.json'; manifest=json.loads(mp.read_text())
    for key, val in dict(model_name='m-a-p/MERT-v1-95M',model_revision='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5',
                         sample_rate=24000,clip_seconds=30.,center_crop=True,
                         pooling='attention-masked mean of final hidden layer',normalization='L2',hidden_size=768).items():
        assert manifest[key]==val, (key, manifest[key])
    ip, vp=cat/'catalog_item_ids.json', cat/'catalog_mert_embeddings_l2.npy'
    for f in [ip,vp]: assert sha(f)==manifest['outputs'][f.name]
    ids=json.loads(ip.read_text()); lookup={str(v):i for i,v in enumerate(ids)}
    assert len(lookup)==len(ids)
    emb=normalized(np.load(vp))
    target=emb[[lookup[r['target']] for r in rows]].reshape(100,5,768)
    history=emb[np.array([[lookup[s] for s in r['history']] for r in rows])]
    dest=O/ds/'oas'; dest.mkdir(parents=True,exist_ok=True)
    out=dict(schema='list5-generation-oas-v1',dataset=ds,lists=100,items_per_list=5,clips_per_arm=500,
             setting='full_catalog',bootstrap=dict(unit='list',repetitions=10000,seed=42),
             definitions=dict(mert_oas='optimal 5x5 matched mean cosine vs five ground-truth target songs; higher better',
                              mert_oas_distance='1 - MERT OAS, equivalent to DDBC cosine hgd_dist; lower better',
                              history_fit='diagnostic only: mean cosine vs original 15 history songs',
                              clap_adherence='own instruction cosine, averaged over five clips; not target OAS',
                              fad='global VGGish distribution distance: all 500 generated clips vs all 500 target occurrences; lower better; no assignment'),
             provenance={str(f):sha(f) for f in [cp,mp,ip,vp]}, arms={},paired={})
    per={}; assignments={}
    for arm in ARMS:
        b=O/ds/arm; system=SYSTEMS[arm]; planpath=b/'render-plan.json'
        plan=json.loads(planpath.read_text()); assert plan['cohort_sha256']==sha(cp)
        for a,r in zip(plan['rows'],rows):
            assert all(a[k]==r[k] for k in r)
        mert=[]
        for row in rows:
            i=row['flat_index']; npz=b/'metrics/mert-rows'/f'{i:04d}.npz'
            q=np.load(npz); identity=json.loads(str(q['identity']))
            assert identity['audio_sha256']==sha(b/'synthesis'/system/f'{i:04d}.mp3')
            assert identity['revision']==manifest['model_revision'] and identity['flat_index']==i
            assert identity['item_position']==i%5 and identity['source_history_index']==row['source_history_index']
            assert identity['sample_rate']==24000 and identity['clip_seconds']==30
            assert identity['pooling']=='attention-masked final-layer mean' and identity['normalization']=='L2'
            mert.append(q['embedding']); out['provenance'][str(npz)]=sha(npz)
        mert=normalized(np.stack(mert)); match=assignment_oas(mert.reshape(100,5,768),target)
        slug=system.lower().replace('-','_')
        ap=b/'metrics/clap'/f'{slug}_clap_audio_l2.npy'; tp=b/'metrics/clap'/f'{slug}_clap_attribute_l2.npy'
        audio,text=normalized(np.load(ap)),normalized(np.load(tp)); assert audio.shape==text.shape and len(audio)==500
        fp=b/'metrics/fad.json'; fad=json.loads(fp.read_text())['fad'][system]
        per[arm]=dict(mert_oas=match['similarity'],mert_oas_distance=match['distance'],
                       history_fit=np.einsum('bd,bkd->bk',mert,history).mean(1).reshape(100,5).mean(1),
                       clap_adherence=np.einsum('bd,bd->b',audio,text).reshape(100,5).mean(1))
        assignments[arm+'__target_positions']=match['target_positions']
        assignments[arm+'__pairwise_cosine']=match['pairwise_cosine']
        out['arms'][arm]=dict(scores={k:summary(v) for k,v in per[arm].items()},fad=fad)
        out['provenance'].update({str(f):sha(f) for f in [planpath,ap,tp,fp]})
    for model in ['ddbc','tiger']:
        for comparator in [model+'_sid','direct','history_random','profile','retrieved']:
            key=model+'_joint minus '+comparator
            out['paired'][key]={k:summary(per[model+'_joint'][k]-per[comparator][k]) for k in per[comparator]}
    np.savez_compressed(dest/'generation-per-list.npz',**assignments,
                        **{arm+'__'+k:v for arm,d in per.items() for k,v in d.items()})
    out['per_list_sha256']=sha(dest/'generation-per-list.npz')
    atomic(dest/'generation-report.json',out)
    print(json.dumps(dict(dataset=ds,arms=out['arms'],paired=out['paired']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True)
    main(p.parse_args().dataset)

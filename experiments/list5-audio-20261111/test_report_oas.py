"""Synthetic temporary fixture: test report wiring, not an experimental result."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import report_oas as m

with tempfile.TemporaryDirectory(prefix='oas-report-test-') as temp:
    root=Path(temp); m.O=root/'output'; ds='mpd'; cat=root/'catalog'; art=root/'art'
    cat.mkdir();art.mkdir();(m.O/ds).mkdir(parents=True)
    m.paths=lambda _:dict(catalog=cat,art=art)
    def put(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
    ids=[str(i) for i in range(5)];put(art/'item_id_to_row.json',{v:i for i,v in enumerate(ids)})
    vectors=np.eye(5,768,dtype=np.float32);np.save(cat/'catalog_mert_embeddings_l2.npy',vectors)
    put(cat/'catalog_item_ids.json',ids)
    revision='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
    put(cat/'mert_manifest.json',dict(model_name='m-a-p/MERT-v1-95M',model_revision=revision,sample_rate=24000,
        clip_seconds=30.,center_crop=True,pooling='attention-masked mean of final hidden layer',normalization='L2',hidden_size=768,
        catalog_mapping_sha256=m.sha(art/'item_id_to_row.json'),outputs={f:m.sha(cat/f) for f in ['catalog_item_ids.json','catalog_mert_embeddings_l2.npy']}))
    rows=[dict(flat_index=i,list_index=i//5,item_position=i%5,source_history_index=i//5,target=str(i%5),history=ids*3) for i in range(500)]
    cp=m.O/ds/'cohort.json';put(cp,dict(rows=rows))
    for arm in m.ARMS:
        b=m.O/ds/arm;system=m.SYSTEMS[arm];put(b/'render-plan.json',dict(cohort_sha256=m.sha(cp),rows=rows))
        (b/'synthesis'/system).mkdir(parents=True);(b/'metrics/mert-rows').mkdir(parents=True)
        for r in rows:
            i=r['flat_index'];ap=b/'synthesis'/system/f'{i:04d}.mp3';ap.write_bytes(b'SYNTHETIC TEST ONLY')
            identity=dict(audio_sha256=m.sha(ap),revision=revision,flat_index=i,item_position=i%5,
                          source_history_index=i//5,sample_rate=24000,clip_seconds=30,
                          pooling='attention-masked final-layer mean',normalization='L2')
            np.savez_compressed(b/'metrics/mert-rows'/f'{i:04d}.npz',embedding=vectors[4-i%5],identity=json.dumps(identity))
        slug=system.lower().replace('-','_');(b/'metrics/clap').mkdir()
        for kind in ['audio','attribute']:np.save(b/'metrics/clap'/f'{slug}_clap_{kind}_l2.npy',np.tile([1.,0.],(500,1)))
        put(b/'metrics/fad.json',{'fad':{system:1.25}})
    with contextlib.redirect_stdout(io.StringIO()):m.main(ds)
    report=json.loads((m.O/ds/'oas/generation-report.json').read_text())
    detail=np.load(m.O/ds/'oas/generation-per-list.npz')
    for arm in m.ARMS:
        scores=report['arms'][arm]['scores']
        assert scores['mert_oas']['mean']==1 and scores['mert_oas_distance']['mean']==0
        assert abs(scores['history_fit']['mean']-.2)<1e-12
        assert scores['clap_adherence']['mean']==1 and report['arms'][arm]['fad']==1.25
        np.testing.assert_array_equal(detail[arm+'__target_positions'],np.tile([4,3,2,1,0],(100,1)))
    assert all(s['mean']==0 for pair in report['paired'].values() for s in pair.values())
    # A missing fifth item must fail rather than silently shrink the denominator.
    (m.O/ds/m.ARMS[0]/'metrics/mert-rows/0004.npz').unlink()
    try:
        with contextlib.redirect_stdout(io.StringIO()):m.main(ds)
    except FileNotFoundError:pass
    else:raise AssertionError('Incomplete five-item group was accepted')
print('PASS: full9-arm x100-list x5-item report fixture, reversed order, FAD preserved, paired deltas, missing-item rejection')

"""Independent exhaustive 5! verification from catalog vectors and item IDs."""
import itertools
import json
import numpy as np
from common import R, O, paths, source_plans, sha, atomic

perms=np.array(list(itertools.permutations(range(5))))
for ds in ['mpd','m4a']:
    x=paths(ds); co=json.loads((O/ds/'cohort.json').read_text())
    v=np.load(x['art']/'catalog_item_embeddings.npy').astype(np.float64)
    v=v/np.linalg.norm(v,axis=1,keepdims=True)
    ids=json.loads((x['art']/'item_id_to_row.json').read_text())
    truth=json.loads((R/'outputs/sampled-candidates-20261110'/ds/'candidates.json').read_text())['truth']
    b=O/ds/'oas'; stored=np.load(b/'retrieval-clhe-per-list.npz')
    report=json.loads((b/'retrieval-clhe-report.json').read_text()); audit={}
    for arm, p in source_plans(ds,'full_catalog').items():
        pred=json.loads(p.read_text())['predictions']['item_ids']; result=[]
        for aa,bb in zip(pred,truth):
            m=v[[ids[z] for z in aa]]@v[[ids[z] for z in bb]].T
            result.append(m[np.arange(5)[None,:],perms].mean(1).max())
        result=np.array(result)
        np.testing.assert_allclose(result,stored[arm+'__similarity'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(1-result,stored[arm+'__distance'],atol=1e-12,rtol=0)
        for name, ix in [('fixed_941_contexts',np.arange(941)),('audio_100_lists',co['indices'])]:
            assert abs(result[ix].mean()-report['scopes'][name]['arms'][arm]['clhe_oas']['mean'])<1e-12
        audit[arm]=dict(n=941,max_absolute_error=float(np.max(np.abs(result-stored[arm+'__similarity']))))
    atomic(b/'independent-verification.json',dict(passed=True,method='All 120 permutations per list from original catalog embeddings and source item IDs; independent of Hungarian implementation',arms=audit,report_sha256=sha(b/'retrieval-clhe-report.json')))
    print(ds,'independent exhaustive verification PASSED')

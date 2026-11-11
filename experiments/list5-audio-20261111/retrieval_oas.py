"""CPU-only CLHE assignment over frozen, already retrieved catalog IDs."""
import argparse
from common import *
from oas import assignment_oas, summary, normalized


def main(ds):
    x = paths(ds)
    art = x['art']; dest = O/ds/'oas'; dest.mkdir(parents=True, exist_ok=True)
    cp = O/ds/'cohort.json'; cohort = json.loads(cp.read_text())
    candpath = R/'outputs/sampled-candidates-20261110'/ds/'candidates.json'
    candidates = json.loads(candpath.read_text())
    manifestpath = art/'wpd_artifact_manifest.json'
    manifest = json.loads(manifestpath.read_text())
    vp, ip = art/'catalog_item_embeddings.npy', art/'item_id_to_row.json'
    for f in [vp, ip]:
        assert sha(f) == manifest['outputs'][f.name]['sha256'], f
    vectors = np.load(vp); mapping = json.loads(ip.read_text())
    assert sorted(mapping.values()) == list(range(len(vectors)))
    normalized(vectors)  # reject corrupt or missing representations, no zero fallback
    truth = candidates['truth']; assert len(truth) == 941
    assert len(cohort['indices']) == 100 and len(set(cohort['indices'])) == 100
    for row in cohort['rows']:
        i, j = row['source_history_index'], row['item_position']
        assert row['target'] == truth[i][j] and row['history'] == candidates['histories'][i]
        assert row['source_test_index'] == candidates['indices'][i]

    def embed(ids):
        ids = np.asarray(ids, dtype=str)
        assert ids.ndim == 2 and ids.shape[1] == 5
        return vectors[np.array([[mapping[s] for s in line] for line in ids])]

    out = dict(schema='list5-clhe-oas-v1', dataset=ds, setting='full_catalog',
               definition='mean cosine under optimal one-to-one 5x5 assignment; higher is better',
               official_ddbc_distance='1 - similarity; lower is better',
               source='https://github.com/LiAi16/DDBC/blob/main/evaluator.py',
               bootstrap=dict(unit='list', repetitions=10000, seed=42),
               catalog_items=len(vectors), embedding_dimension=vectors.shape[1],
               embedding_source=manifest['source'],
               provenance={str(f): sha(f) for f in [cp, candpath, manifestpath, vp, ip]},
               scopes={})
    arrays = {}; results = {}; source = source_plans(ds, 'full_catalog')
    for arm, planpath in source.items():
        plan = json.loads(planpath.read_text()); pred = plan['predictions']
        assert pred['target_item_ids'] == truth and len(pred['item_ids']) == 941
        if 'source_test_indices' in plan:
            assert plan['source_test_indices'] == candidates['indices']
        rr = json.loads((O/ds/arm/'render-plan.json').read_text())['rows']
        assert [r['predicted_item_id'] for r in rr] == [pred['item_ids'][r['source_history_index']][r['item_position']] for r in rr]
        result = assignment_oas(embed(pred['item_ids']), embed(truth))
        results[arm] = result
        arrays.update({arm+'__'+k:v for k,v in result.items()})
        out['provenance'][str(planpath)] = sha(planpath)
    for name, idx in [('fixed_941_contexts', np.arange(941)), ('audio_100_lists', np.array(cohort['indices']))]:
        scope = dict(n_lists=len(idx), arms={}, paired={})
        for arm, res in results.items():
            scope['arms'][arm] = dict(clhe_oas=summary(res['similarity'][idx]),
                                     official_cosine_distance=summary(res['distance'][idx]))
        for model in ['ddbc', 'tiger']:
            delta = results[model+'_joint']['similarity'][idx] - results[model+'_sid']['similarity'][idx]
            scope['paired'][model+'_joint minus '+model+'_sid'] = summary(delta)
        out['scopes'][name] = scope
    # Renderer controls share a retrieval only if the actual item IDs prove it.
    scope = out['scopes']['audio_100_lists']
    for arm in ['drop', 'retrieved', 'direct', 'history_random', 'profile']:
        pp = O/ds/arm/'render-plan.json'; rows = json.loads(pp.read_text())['rows']
        out['provenance'][str(pp)] = sha(pp)
        ids = [r['predicted_item_id'] for r in rows]
        if all(s is None for s in ids):
            scope['arms'][arm] = dict(clhe_oas=None, reason='Pure generation: no retrieved catalog item IDs')
            continue
        assert all(s is not None for s in ids)
        res = assignment_oas(embed(np.array(ids).reshape(100,5)), embed(np.array([r['target'] for r in rows]).reshape(100,5)))
        scope['arms'][arm] = dict(clhe_oas=summary(res['similarity']), official_cosine_distance=summary(res['distance']))
        arrays.update({arm+'__audio100__'+k:v for k,v in res.items()})
    arrays['audio100_source_indices'] = np.array(cohort['indices'])
    np.savez_compressed(dest/'retrieval-clhe-per-list.npz', **arrays)
    out['per_list_sha256'] = sha(dest/'retrieval-clhe-per-list.npz')
    atomic(dest/'retrieval-clhe-report.json', out)
    print(json.dumps(dict(dataset=ds, scopes=out['scopes']), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=['mpd','m4a'], required=True)
    main(p.parse_args().dataset)

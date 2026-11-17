"""CPU-only simple baselines on frozen cohorts; no model fitting or seed selection."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predictions(method, catalog, histories, popularity, seed, pools=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i, history in enumerate(histories):
        pool = catalog if pools is None else pools[i]
        allowed = set(pool)
        if method == 'uniform_random':
            row = rng.choice(pool, 5, replace=False).tolist()
        else:
            prefix = list(reversed(history)) if method == 'recent_history' else []
            row = list(dict.fromkeys(x for x in prefix + popularity if x in allowed))[:5]
        assert len(row) == len(set(row)) == 5 and set(row) <= allowed
        rows.append(row)
    return rows


def basic_score(rows, truth, histories, itemc, catalog):
    recall, cue_f1, repeat, hit = [], [], [], []
    for pred, target, hist in zip(rows, truth, histories):
        overlap = sum((Counter(pred) & Counter(target)).values())
        recall.append(overlap / 5)
        hit.append(float(overlap > 0))
        a = Counter(c for s in pred for c in itemc[s][:8])
        b = Counter(c for s in target for c in itemc[s][:8])
        assert sum(a.values()) == sum(b.values()) == 40
        cue_f1.append(sum((a & b).values()) / 40)
        repeat.append(sum(x in set(hist) for x in pred) / 5)
    per = dict(recall_at_5=recall, cue_f1=cue_f1, hit_at_5=hit, history_repeat_rate=repeat)
    totals = {k: float(np.mean(v)) for k, v in per.items()}
    totals['diversity'] = len({s for row in rows for s in row}) / len(catalog)
    return totals, per


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main(workspace):
    from datasets import load_from_disk
    root = Path(workspace).resolve()
    output = root / 'outputs/GenPlaylist-simple-baselines-20261112'
    evidence = root / 'work/sampled-candidates-20261110/complete-evidence'
    oas_path = Path(__file__).parents[1] / 'list5-audio-20261111/oas.py'
    spec = importlib.util.spec_from_file_location('simple_baseline_oas', oas_path)
    oas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oas)
    protocol = dict(
        datasets=['mpd', 'm4a'], contexts_per_dataset=941, output_items=5,
        scopes=['fixed_941_contexts', 'audio_100_lists'], random_seeds=list(range(10)),
        generation_seed=0, random_rule='Uniform from eligible catalog without replacement; history items allowed',
        popularity_rule='Descending counts in training target positions; catalog-row tie break; history items allowed',
        recent_rule='Most recent five distinct eligible history items; training popularity fills any shortage',
        cues='Retrieved first8 catalog cues per selected item; no learned cue prediction',
        diversity='Unique recommended catalog IDs / catalog size; not audio diversity',
        settings=['full_catalog', 'sampled100_diagnostic'],
        sampled_policy='Reuse frozen100negatives+unique_targets pool; never compare sampled scores to full-catalog scores',
        uncertainty='Random: mean and sample SD across10fixedseeds, not confidence intervals; deterministic: single result',
        audio_status='No new audio rendered; selected seed0 input plans saved for same100listsx5 downstream rendering',
        source_sha256=sha(__file__), oas_source_sha256=sha(oas_path))
    write(output/'protocol.json', protocol)
    result = {}
    for ds in ['mpd', 'm4a']:
        candpath = evidence/ds/'candidates.json'
        cand = json.loads(candpath.read_text())
        cp = evidence/f'{ds}-item2cues.json'
        itemc = json.loads(cp.read_text())
        prep = root/'work/migration/original-ncl/data/processed'/(
            'genplaylist-v4-8cue-20item-joint-15to5' if ds == 'mpd' else
            'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
        data = load_from_disk(str(prep/'raw_dataset'))
        test = data['test'].select(cand['indices'])
        seq = [[str(s) for s in x] for x in test['item_seq']]
        assert [x[:15] for x in seq] == cand['histories']
        assert [x[15:] for x in seq] == cand['truth']
        counts = Counter(str(s) for row in data['train']['item_seq'] for s in row[15:])
        ids = cand['catalog_ids']; order = {s: i for i, s in enumerate(ids)}
        popularity = sorted(ids, key=lambda s: (-counts[s], order[s]))
        write(output/ds/'train-popularity-counts.json', {s: counts[s] for s in ids})
        art = root/('work/migration/local-dataset' if ds == 'mpd' else
                    'work/migration/original-ncl/data/artifacts-music4all-onion-v3-u500')
        vp, ip = art/'catalog_item_embeddings.npy', art/'item_id_to_row.json'
        manifest = json.loads((art/'wpd_artifact_manifest.json').read_text())
        for p in [vp, ip]:
            assert sha(p) == manifest['outputs'][p.name]['sha256']
        emb = np.load(vp); mapping = json.loads(ip.read_text())
        def embed(rows):
            return emb[np.array([[mapping[s] for s in row] for row in rows])]
        targets = embed(cand['truth'])
        cohortpath = root/'outputs/GenPlaylist-list5-audio-20261111'/ds/'cohort.json'
        cohort = json.loads(cohortpath.read_text()); subset = cohort['indices']
        assert len(subset) == len(set(subset)) == 100
        scopes = {'fixed_941_contexts': np.arange(941), 'audio_100_lists': np.array(subset)}
        report = dict(dataset=ds, catalog_size=len(ids), settings={},
                      provenance={str(p): sha(p) for p in [candpath, cp, vp, ip, cohortpath]},
                      expected_full_catalog_random_recall=float(np.mean([len(set(t))/len(ids) for t in cand['truth']])),
                      expected_sampled_random_recall=float(np.mean([len(set(t))/len(pool) for t, pool in zip(cand['truth'],cand['candidates'])])))
        for setting in protocol['settings']:
            pools = None if setting == 'full_catalog' else cand['candidates']
            methods = {}
            for method in ['uniform_random', 'train_popularity', 'recent_history']:
                runs = []
                for seed in (protocol['random_seeds'] if method == 'uniform_random' else [0]):
                    rows = predictions(method, ids, cand['histories'], popularity, seed, pools)
                    scores, per = basic_score(rows, cand['truth'], cand['histories'], itemc, ids)
                    match = oas.assignment_oas(embed(rows), targets)
                    perms = np.array(list(itertools.permutations(range(5))))
                    brute = [m[np.arange(5)[None, :], perms].mean(1).max() for m in match['pairwise_cosine'][:10]]
                    np.testing.assert_allclose(brute, match['similarity'][:10], atol=1e-12)
                    per['clhe_oas'] = match['similarity'].tolist()
                    scope_scores = {}
                    for scope, ix in scopes.items():
                        scope_scores[scope] = {k: float(np.mean(np.asarray(v)[ix])) for k,v in per.items()}
                        scope_scores[scope]['diversity'] = len({s for i in ix for s in rows[i]})/len(ids)
                    runs.append(dict(seed=seed, scopes=scope_scores))
                    write(output/ds/setting/method/f'seed{seed}.json', dict(
                        dataset=ds, setting=setting, method=method, seed=seed,
                        source_test_indices=cand['indices'], predictions=dict(item_ids=rows,
                        cue_ids=[[itemc[s][:8] for s in row] for row in rows], target_item_ids=cand['truth']),
                        metrics=scope_scores, per_context=per))
                    if setting == 'full_catalog' and seed == 0:
                        write(output/ds/'audio-inputs'/f'{method}.json', dict(
                            method=method, seed=seed, source_history_indices=subset,
                            source_test_indices=[cand['indices'][i] for i in subset],
                            histories=[cand['histories'][i] for i in subset],
                            item_ids=[rows[i] for i in subset],
                            cue_ids=[[itemc[s][:8] for s in rows[i]] for i in subset],
                            audio_rendered=False))
                methods[method] = dict(runs=runs, summary={scope: {metric: dict(
                    mean=float(np.mean([r['scopes'][scope][metric] for r in runs])),
                    seed_sd=float(np.std([r['scopes'][scope][metric] for r in runs],ddof=1)) if len(runs)>1 else None)
                    for metric in runs[0]['scopes'][scope]} for scope in scopes})
            report['settings'][setting] = methods
        result[ds] = report
        write(output/ds/'report.json', report)
        print(ds, 'completed', flush=True)
    write(output/'report.json', result)
    write(output/'complete.json', dict(datasets=2, methods=3, settings=2, random_seeds=10,
        contexts_per_case=941, prediction_runs=48, audio_generated=0, report_sha256=sha(output/'report.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--workspace', required=True)
    main(p.parse_args().workspace)

"""Bounded, CPU-only provenance check of repaired Stage A candidates; no generation."""
import collections
import datetime
import json
import sys
import time
from pathlib import Path

R = Path('/storage_fast/ttu/GenPlaylist')
S = R / 'outputs/cue-v2-stageA-20261112'
O = R / 'outputs/cue-v2-repair-r5-stageA-20261113'
sys.path.insert(0, str(R / 'experiments/cue-v2-repair-r3-20261113'))
from pilot_extract import atomic, evidence_units, paths, sha
from extract import validate_candidate


def main():
    deadline = datetime.datetime.fromisoformat('2026-11-13T11:11:30+08:00').timestamp()
    while not (O / 'candidates-complete.json').exists():
        if (O / 'failure.json').exists():
            raise RuntimeError('Candidate producer failed; preserve outputs for inspection')
        if time.time() >= deadline:
            atomic(O / 'verification-wait-ended.json', {'reason': 'deadline', 'complete': False})
            return
        time.sleep(30)
    config = json.loads((O / 'run_config.json').read_text())
    reports = {}
    manifest = {}
    for ds in ['mpd', 'm4a']:
        cohort = json.loads((S / ds / 'cohort.json').read_text())
        inputs = json.loads((S / ds / 'inputs.json').read_text())
        meta = json.loads((paths(ds)['data'] / 'catalog_metadata.json').read_text())
        assert sha(S / ds / 'cohort.json') == config['source'][ds]['cohort_sha256']
        assert sha(S / ds / 'inputs.json') == config['source'][ds]['inputs_sha256']
        counts = collections.Counter()
        for i in cohort['extraction_ids']:
            phases = {}
            for phase in ['pass1', 'pass2', 'audited']:
                p = O / phase / ds / (i + '.json')
                q = phases[phase] = json.loads(p.read_text())
                assert q['input'] == inputs[i], (ds, i, phase, 'input mismatch')
                manifest[str(p.relative_to(O))] = sha(p)
                if 'reused_stageA_pilot_source' in q:
                    assert sha(q['reused_stageA_pilot_source']) == q['reused_source_sha256']
                units = evidence_units(inputs[i])
                for c in q['accepted']:
                    assert c['evidence_id'] in units
                    unit = units[c['evidence_id']]
                    assert c['source'] == unit['source'] and c['evidence'] == unit['text']
                    assert validate_candidate(c, inputs[i], meta[i]) is None, (ds, i, c)
            assert all(c in phases['pass2']['accepted'] for c in phases['audited']['accepted'])
            audited = phases['audited']
            assert len(audited['accepted']) + len(audited.get('semantic_rejected', [])) == len(phases['pass2']['accepted'])
            counts['songs'] += 1
            counts['first_pass_accepted'] += len(phases['pass1']['accepted'])
            counts['completion_accepted'] += len(phases['pass2']['accepted'])
            counts['after_semantic_filter'] += len(audited['accepted'])
            if inputs[i]['lyrics']:
                counts['lyric_songs'] += 1
                counts['unique_candidate_capacity_slots'] += min(8, len({c['phrase'] for c in audited['accepted']}))
        expected = {i + '.json' for i in cohort['extraction_ids']}
        for phase in ['pass1', 'pass2', 'audited']:
            assert {p.name for p in (O / phase / ds).glob('*.json')} == expected
        reports[ds] = dict(counts=counts, candidate_capacity=counts['unique_candidate_capacity_slots'] / (8 * counts['lyric_songs']))
    assert sum(v['counts']['songs'] for v in reports.values()) == 3544
    atomic(O / 'candidate-provenance-manifest.json', manifest)
    atomic(O / 'candidate-independent-verification.json', dict(
        passed=True, datasets=reports, config_sha256=sha(O / 'run_config.json'),
        manifest_sha256=sha(O / 'candidate-provenance-manifest.json'),
        meaning='Source, hard-filter, subset and capacity checks only; same-family semantic review is not human ground truth.',
        final_mapping_coverage_not_tested=True, repaired_audio_started=False, stage_B_started=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        atomic(O / 'candidate-verification-failure.json', {'traceback': traceback.format_exc()})
        raise

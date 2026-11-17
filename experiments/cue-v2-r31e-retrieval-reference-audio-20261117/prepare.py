"""Freeze no-cue, five-retrieved-song reference plans for audio generation."""
import argparse
import json

from common import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
args = parser.parse_args()
dataset = args.dataset
candidate_path = R / 'outputs/sampled-candidates-20261110' / dataset / 'candidates.json'
candidates = json.loads(candidate_path.read_text())
indices = sample_indices(dataset)
plans = {name: json.loads(path.read_text()) for name, path in source_plans(dataset).items()}
for name, plan in plans.items():
    prediction = plan['predictions']
    assert len(prediction['item_ids']) == 941
    assert prediction['target_item_ids'] == candidates['truth'], name
    assert all(len(row) == 5 and len(set(row)) == 5 for row in prediction['item_ids']), name

cohort = {
    'schema': 'cue-v2-r31e-retrieval-reference-cohort-v1',
    'dataset': dataset,
    'candidate_setting': 'full_catalog',
    'lists': 100,
    'outputs_per_list': 5,
    'renderings_per_arm': 500,
    'sampling_seed': 20261111,
    'source_cohort_sha256': sha(candidate_path),
    'source_history_indices': indices,
    'source_test_indices': [candidates['indices'][index] for index in indices],
}
immutable(O / dataset / 'cohort.json', cohort)

for arm, source in source_plans(dataset).items():
    prediction = plans[arm]['predictions']['item_ids']
    rows = []
    for list_index, history_index in enumerate(indices):
        references = [str(item) for item in prediction[history_index]]
        for position in range(5):
            rows.append({
                'flat_index': list_index * 5 + position,
                'list_index': list_index,
                'source_history_index': history_index,
                'source_test_index': candidates['indices'][history_index],
                'item_position': position,
                'reference_item_ids': references,
                'target_item_id': candidates['truth'][history_index][position],
                'seed': clip_seed(history_index, position),
            })
    assert len(rows) == 500
    plan = {
        'schema': 'cue-v2-r31e-retrieval-reference-render-plan-v1',
        'dataset': dataset,
        'arm': arm,
        'cohort_sha256': sha(O / dataset / 'cohort.json'),
        'source_plan': str(source),
        'source_plan_sha256': sha(source),
        'cue_conditioning': 'none',
        'reference_conditioning': 'retrieved_top5_only',
        'rows': rows,
    }
    immutable(O / dataset / arm / 'render-plan.json', plan)
print(json.dumps({'dataset': dataset, 'arms': list(plans), 'clips_per_arm': 500}, indent=2))

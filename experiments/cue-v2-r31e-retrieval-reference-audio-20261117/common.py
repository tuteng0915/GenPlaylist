import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path('/storage_fast/ttu/GenPlaylist')
E = R / 'experiments/cue-v2-r31e-retrieval-reference-audio-20261117'
O = R / 'outputs/cue-v2-r31e-retrieval-reference-audio-20261117'
CODE = R / 'v2-recovery-20261106/code'
ARMS = ('ddbc_reference', 'tiger_reference')
SYSTEM = 'ACE-Step-Direct'
REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, str(path)
    else:
        atomic(path, value)


def module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(specification)
    sys.modules[name] = loaded
    specification.loader.exec_module(loaded)
    return loaded


def paths(dataset):
    return module(
        'retrieval_reference_paths',
        R / 'experiments/history-cue-baselines-20261110/common.py',
    ).paths(dataset)


def sample_indices(dataset):
    seed = 1 if dataset == 'mpd' else 2
    return np.sort(np.random.default_rng(np.random.SeedSequence([20261111, seed])).choice(941, 100, replace=False)).tolist()


def clip_seed(history_index, position):
    return 42000 + history_index + 941 * position


def source_plans(dataset):
    values = {
        'ddbc_reference': (
            R / 'experiments/two-dataset-fair-20261108/mpd-existing-plans/sid_matched.json'
            if dataset == 'mpd'
            else R / 'outputs/two-dataset-fair-20261108/m4a/plans/sid_matched-audio.json'
        ),
    }
    if dataset == 'mpd':
        # This plan is materialized only after the corrected SID-only MPD
        # control finishes; never fall back to the audited-invalid old plan.
        values['tiger_reference'] = (
            R / 'outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/mpd/sid/audio/plan.json'
        )
    else:
        values['tiger_reference'] = (
            R / 'outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/m4a/sid/audio/plan.json'
        )
    return values

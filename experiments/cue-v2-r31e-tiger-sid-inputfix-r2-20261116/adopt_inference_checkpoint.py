"""Adopt a completed TIGER checkpoint after an inference-only code fix.

The source checkpoint's model, optimizer, RNG, and training provenance are
unchanged.  The target namespace differs solely because its `core.py` fixes a
Torch conversion in SID-only inference; this script records the parent hash
instead of overwriting the source artifact.
"""
import hashlib
import json
import os
from pathlib import Path

import torch


SOURCE = Path('/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-tiger-sid-inputfix-r1-20261116/m4a/sid/last.ckpt')
TARGET = Path('/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/m4a/sid/last.ckpt')
RECIPE = TARGET.with_name('recipe.json')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if TARGET.exists():
        print('EXISTS', TARGET)
        return
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    recipe = json.loads(RECIPE.read_text())
    assert source['step'] == 20_000
    assert source['recipe']['dataset'] == recipe['dataset'] == 'm4a'
    assert source['recipe']['arm'] == recipe['arm'] == 'sid'
    assert source['recipe']['encoder'] == recipe['encoder']
    assert source['recipe']['core_sha256'] != recipe['core_sha256']
    source['recipe'] = recipe
    source['inference_code_fix_parent'] = {
        'path': str(SOURCE),
        'sha256': sha256(SOURCE),
        'reason': 'SID-only inference converts catalog SID table to torch before device transfer; training state unchanged',
    }
    temp = TARGET.with_suffix('.tmp')
    torch.save(source, temp)
    os.replace(temp, TARGET)
    print(json.dumps({'target': str(TARGET), 'sha256': sha256(TARGET),
                      'parent': source['inference_code_fix_parent']}, indent=2))


if __name__ == '__main__':
    main()

"""Synthetic end-to-end test for the one-time server migration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def test_prepare_with_explicit_source_mapping():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        legacy = root / "legacy"
        output = root / "output"
        legacy.mkdir()
        catalog = {
            "18996": {"item_id": "wrong-inner-value", "title": "A"},
            "48262": {"item_id": "48262", "title": "B"},
        }
        (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        source = np.stack([
            np.full(64, 10.0, dtype=np.float32),
            np.full(64, 20.0, dtype=np.float32),
            np.full(64, 30.0, dtype=np.float32),
        ])
        np.save(legacy / "items.npy", source)
        np.save(legacy / "clhe_weight.npy", np.zeros((768, 64), dtype=np.float32))
        (legacy / "clhe_token.json").write_text(json.dumps({
            "18996": [1, 257, 513, 769],
            "48262": [256, 512, 768, 842],
        }), encoding="utf-8")
        (legacy / "source_mapping.json").write_text(json.dumps({
            "18996": 2, "48262": 0,
        }), encoding="utf-8")

        subprocess.run([
            sys.executable, str(HERE / "prepare_server_artifacts.py"),
            "--legacy-dir", str(legacy),
            "--catalog", str(root / "catalog.json"),
            "--output-dir", str(output),
            "--source-embeddings", str(legacy / "items.npy"),
            "--source-item-id-to-row", str(legacy / "source_mapping.json"),
        ], check=True, capture_output=True, text=True)

        selected = np.load(output / "catalog_item_embeddings.npy")
        assert selected.shape == (2, 64)
        assert np.allclose(selected[0], 30.0)
        assert np.allclose(selected[1], 10.0)
        mapping = json.loads((output / "item_id_to_row.json").read_text())
        assert mapping == {"18996": 0, "48262": 1}
        manifest = json.loads((output / "wpd_artifact_manifest.json").read_text())
        assert manifest["selection_mode"] == "explicit-source-mapping"


if __name__ == "__main__":
    test_prepare_with_explicit_source_mapping()
    print("  PASS  test_prepare_with_explicit_source_mapping")

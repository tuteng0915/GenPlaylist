#!/usr/bin/env python3
"""Unit tests for the frozen end-to-end ACE-Step runner helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("run_end_to_end_synthesis.py")
SPEC = importlib.util.spec_from_file_location("end_to_end_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        snapshot = root / "snapshots" / "frozen-revision"
        for name in ("ace_step_transformer", "music_dcae_f8c8", "music_vocoder", "umt5-base"):
            (snapshot / name).mkdir(parents=True)
        assert MODULE._model_revision(snapshot) == "frozen-revision"

        verbalizations = root / "verbalizations"
        record_path = verbalizations / "GenPlaylist" / "0007.json"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps({
            "system": "GenPlaylist",
            "example_index": 7,
            "music_attributes": "indie pop, 100 BPM, English",
            "lyric_draft": "[verse]\nA new line",
        }), encoding="utf-8")
        loaded = MODULE._load_verbalization(verbalizations, "GenPlaylist", 7)
        assert loaded["example_index"] == 7

        bad = dict(loaded)
        bad["lyric_draft"] = ""
        record_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            MODULE._load_verbalization(verbalizations, "GenPlaylist", 7)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty lyrics were accepted")
    print("end-to-end synthesis helper tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

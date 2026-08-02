"""Smoke test for src/eval_matrix.evaluate.

Loads a curated playlist JSON and runs the evaluation matrix using the seed
songs as the (stand-in) generated set and the ground-truth songs as the
reference set. This exercises the full CLAP + FAD path end to end; it is not a
meaningful quality measurement (seed songs are real, not model output).
"""

import json
import os

from eval_matrix import evaluate

# Resolve paths relative to the repo root so the script works from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATED_JSON = os.path.join(REPO_ROOT, "data", "curated", "p6334_v1.json")


def main():
    with open(CURATED_JSON) as f:
        playlist = json.load(f)

    generated = [
        os.path.join(REPO_ROOT, s["audio_path"]) for s in playlist["seed_songs"]
    ]
    ground_truth = [
        os.path.join(REPO_ROOT, s["audio_path"])
        for s in playlist["ground_truth_songs"]
    ]

    scores = evaluate(
        generated_audio_paths=generated,
        ground_truth_audio_paths=ground_truth,
        prompt_text="energetic upbeat pop",
    )

    print(scores)


if __name__ == "__main__":
    main()

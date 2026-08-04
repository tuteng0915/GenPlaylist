"""Corpus-level evaluation for generated vs ground-truth next songs.

Implements the two encoder-based metrics kept for WP-D Part 4:

  * CLAP_OAS  — Optimal Assignment Similarity. Cosine similarity matrix between
                generated and ground-truth CLAP audio embeddings, solved as a
                minimum-cost bipartite matching (Hungarian algorithm, cost =
                1 - similarity), then the matched similarities are averaged.
                This is the DDBC-style optimal-matching / OAS score restricted
                to the CLAP embedding space.
  * FAD       — Frechet Audio Distance between all generated next songs and
                all held-out ground-truth next songs in an evaluation corpus.
  * CLAP_Sim  — Condition-adherence check: mean cosine similarity between each
                generated song's CLAP audio embedding and the CLAP text
                embedding of the prompt (only when ``prompt_text`` is given).

Only CLAP (laion_clap) and FAD are used here — no MERT or ImageBind.
"""

import os
import tempfile
import shutil

import numpy as np


def _l2_normalize(x, axis=-1, eps=1e-8):
    """Row-wise L2 normalization so dot products give cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def _load_clap():
    """Load a CLAP model with the default (non-fusion) checkpoint."""
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # downloads/loads the default 630k-audioset checkpoint
    return model


def _clap_audio_embeddings(model, audio_paths):
    """Return an (N, D) array of CLAP audio embeddings for the given files."""
    emb = model.get_audio_embedding_from_filelist(x=list(audio_paths), use_tensor=False)
    return np.asarray(emb, dtype=np.float32)


def _clap_text_embedding(model, text):
    """Return a (1, D) CLAP text embedding for a single prompt string."""
    # laion_clap expects a list; passing a single item can trip batchnorm, so
    # duplicate the prompt and keep the first row.
    emb = model.get_text_embedding([text, text], use_tensor=False)
    return np.asarray(emb, dtype=np.float32)[:1]


def _clap_oas(gen_emb, gt_emb):
    """Optimal-assignment similarity via Hungarian matching on 1 - cosine."""
    from scipy.optimize import linear_sum_assignment

    gen_n = _l2_normalize(gen_emb)
    gt_n = _l2_normalize(gt_emb)
    if gen_n.ndim != 2 or gt_n.ndim != 2 or not len(gen_n) or not len(gt_n):
        raise ValueError("CLAP OAS requires two non-empty 2-D embedding matrices")
    if gen_n.shape[1] != gt_n.shape[1]:
        raise ValueError(
            f"CLAP embedding dimensions differ: {gen_n.shape[1]} vs {gt_n.shape[1]}")
    sim = gen_n @ gt_n.T  # (n_gen, n_gt) cosine similarity matrix
    cost = 1.0 - sim
    row_idx, col_idx = linear_sum_assignment(cost)
    matched = sim[row_idx, col_idx]
    return float(matched.mean())


def _symlink_into_tmpdir(audio_paths):
    """Create a temp dir of symlinks to the given files (FAD needs a dir)."""
    tmp = tempfile.mkdtemp(prefix="fad_")
    for i, p in enumerate(audio_paths):
        src = os.path.abspath(p)
        # Prefix with the index to guarantee unique names across playlists.
        dst = os.path.join(tmp, f"{i:03d}_{os.path.basename(p)}")
        os.symlink(src, dst)
    return tmp


def _compute_fad(gen_paths, gt_paths):
    """Frechet Audio Distance between the generated and ground-truth sets."""
    from frechet_audio_distance import FrechetAudioDistance

    fad = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )
    gt_dir = _symlink_into_tmpdir(gt_paths)
    gen_dir = _symlink_into_tmpdir(gen_paths)
    try:
        score = fad.score(background_dir=gt_dir, eval_dir=gen_dir)
    finally:
        shutil.rmtree(gt_dir, ignore_errors=True)
        shutil.rmtree(gen_dir, ignore_errors=True)
    return float(score)


def evaluate(generated_audio_paths, ground_truth_audio_paths, prompt_text=None):
    """Score aligned generated and ground-truth next-song collections.

    Args:
        generated_audio_paths: iterable of audio file paths for the generated
            next songs, one per reference sequence.
        ground_truth_audio_paths: iterable of audio file paths for the
            held-out next songs in the same example order.
        prompt_text: optional text prompt / creative cue. When provided,
            CLAP_Sim measures how well the generated audio adheres to it.

    Returns:
        dict with keys CLAP_OAS, FAD, CLAP_Sim (CLAP_Sim is None when no
        prompt_text is given).
    """
    generated_audio_paths = list(generated_audio_paths)
    ground_truth_audio_paths = list(ground_truth_audio_paths)
    if not generated_audio_paths or not ground_truth_audio_paths:
        raise ValueError("Generated and ground-truth audio sets must both be non-empty")

    for p in generated_audio_paths + ground_truth_audio_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Audio file not found: {p}")

    model = _load_clap()
    gen_emb = _clap_audio_embeddings(model, generated_audio_paths)
    gt_emb = _clap_audio_embeddings(model, ground_truth_audio_paths)

    scores = {
        "CLAP_OAS": _clap_oas(gen_emb, gt_emb),
        "FAD": _compute_fad(generated_audio_paths, ground_truth_audio_paths),
        "CLAP_Sim": None,
    }

    if prompt_text is not None:
        txt_emb = _clap_text_embedding(model, prompt_text)
        gen_n = _l2_normalize(gen_emb)
        txt_n = _l2_normalize(txt_emb)
        sims = gen_n @ txt_n.T  # (n_gen, 1) cosine similarity to the prompt
        scores["CLAP_Sim"] = float(sims.mean())

    return scores

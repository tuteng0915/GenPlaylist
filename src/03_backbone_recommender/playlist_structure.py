"""playlist_structure.py — Playlist centroid and dispersion computation.

Implements Eq.(1) from the GenPlaylist paper:

    μ_C  = (1/|C|) Σ E(m)
    σ²_C = (1/|C|) Σ ||E(m) - μ_C||²

These two scalars are the conditioning signals fed into the diffusion model
(alongside the noise-level τ) via AdaLN in every Transformer block.

TODOs
-----
- [ ] Switch from generic CLHE proxy (faiss weight matrix) to real CLHE encoder output
      once CLHE embeddings are available for both datasets
- [ ] Precompute and cache (μ_C, σ²_C) per playlist split to avoid redundant computation
      during training — store alongside tokenized dataset
- [ ] Compute Q33 / Q66 of σ²_C over training set and save to dataset dir for
      compact/medium/diverse tier analysis (Table 3 in paper)
- [ ] Expose a batch version: compute_playlist_structure_batch for DataLoader integration
"""

import numpy as np
import torch
from typing import Union


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_playlist_structure(
    item_ids: list,
    emb_matrix: Union[np.ndarray, torch.Tensor],
) -> tuple:
    """Compute centroid μ_C and dispersion σ²_C for a playlist prefix.

    Parameters
    ----------
    item_ids:
        List of integer item IDs in the playlist context C.
    emb_matrix:
        Shape (N_items, d) — embedding matrix for all catalog items.
        Row i corresponds to item_id i.
        Currently a faiss-derived weight matrix; swap for CLHE output later.

    Returns
    -------
    (mu_c, sigma_c2)
        mu_c:     np.ndarray of shape (d,) — playlist centroid.
        sigma_c2: float — playlist dispersion scalar.
    """
    if isinstance(emb_matrix, torch.Tensor):
        emb_matrix = emb_matrix.detach().cpu().numpy()

    ids = [int(i) for i in item_ids]
    embs = emb_matrix[ids]              # (|C|, d)

    mu_c = embs.mean(axis=0)            # (d,)
    diffs = embs - mu_c[None, :]        # (|C|, d)
    sigma_c2 = float((diffs ** 2).sum(axis=1).mean())   # scalar

    return mu_c, sigma_c2


def compute_playlist_structure_batch(
    item_ids_batch: list,
    emb_matrix: Union[np.ndarray, torch.Tensor],
) -> tuple:
    """Batch version: compute (μ_C, σ²_C) for a list of playlist prefixes.

    Parameters
    ----------
    item_ids_batch:
        List of lists of item IDs — one per playlist in the batch.
        Playlists may have different lengths; padding is NOT applied.
    emb_matrix:
        Shape (N_items, d).

    Returns
    -------
    (mu_c_batch, sigma_c2_batch)
        mu_c_batch:     list of np.ndarray (d,), one per playlist.
        sigma_c2_batch: list of float, one per playlist.
    """
    mu_c_batch = []
    sigma_c2_batch = []
    for item_ids in item_ids_batch:
        mu_c, sigma_c2 = compute_playlist_structure(item_ids, emb_matrix)
        mu_c_batch.append(mu_c)
        sigma_c2_batch.append(sigma_c2)
    return mu_c_batch, sigma_c2_batch


# ---------------------------------------------------------------------------
# Dispersion tier thresholds
# ---------------------------------------------------------------------------

def compute_dispersion_tiers(sigma_c2_list: list) -> dict:
    """Compute Q33 and Q66 thresholds from training dispersion values.

    Parameters
    ----------
    sigma_c2_list:
        List of σ²_C values computed over all training playlists.

    Returns
    -------
    dict with keys "q33" and "q66" — used to partition test playlists
    into compact / medium / diverse tiers for Table 3.
    """
    arr = np.array(sigma_c2_list)
    q33 = float(np.percentile(arr, 33))
    q66 = float(np.percentile(arr, 66))
    return {"q33": q33, "q66": q66}


def get_dispersion_tier(sigma_c2: float, q33: float, q66: float) -> str:
    """Return 'compact', 'medium', or 'diverse' for a given σ²_C."""
    if sigma_c2 < q33:
        return "compact"
    elif sigma_c2 < q66:
        return "medium"
    else:
        return "diverse"

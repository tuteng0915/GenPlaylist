"""Checkpoint-backed WP-D adapter used by ``pipeline.genplaylist``.

All heavyweight imports and model loading are lazy. Local development can
therefore import the full pipeline without PyTorch, CUDA, or a checkpoint.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import numpy as np

WP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WP_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(WP_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from genplaylist_tokenizer import GenPlaylistTokenizer  # noqa: E402

_RUNTIME_CACHE = None


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _compose_config(torch):
    import hydra
    from omegaconf import OmegaConf

    resolvers = {
        "cwd": lambda: os.getcwd(),
        "device_count": lambda: max(torch.cuda.device_count(), 1),
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    overrides = shlex.split(os.environ.get("GENPLAYLIST_CONFIG_OVERRIDES", ""))
    with hydra.initialize_config_dir(
            version_base=None, config_dir=str(WP_ROOT / "configs")):
        return hydra.compose(config_name="config", overrides=overrides)


def _load_runtime(catalog_items, catalog_embs, item_id_to_row):
    global _RUNTIME_CACHE
    checkpoint = _path_from_env(
        "GENPLAYLIST_BACKBONE_CKPT", REPO_ROOT / "checkpoints" / "genplaylist-v1.ckpt")
    if _RUNTIME_CACHE is not None:
        cached_checkpoint, model, tokenizer, torch = _RUNTIME_CACHE
        if cached_checkpoint != checkpoint:
            raise RuntimeError(
                "GENPLAYLIST_BACKBONE_CKPT changed after the model was loaded; restart the process")
        return model, tokenizer, torch
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing GenPlaylist-v1 checkpoint: {checkpoint}. "
            "Set GENPLAYLIST_BACKBONE_CKPT to the newly trained checkpoint.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The WP-D runtime requires PyTorch") from exc
    from diffusion import Diffusion

    config = _compose_config(torch)
    data_dir = _path_from_env("GENPLAYLIST_DATASET_DIR", REPO_ROOT / "data" / "dataset")
    cue_dir = _path_from_env(
        "GENPLAYLIST_CUE_DIR",
        SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest")
    tokenizer = GenPlaylistTokenizer.from_files(
        semantic_tokens_path=data_dir / "semantic_tokens.json",
        item2cues_path=cue_dir / "item2cues.json",
        cue_manifest_path=cue_dir / "cue_manifest.json",
        catalog_items=catalog_items,
        catalog_embeddings=catalog_embs,
        item_id_to_row=item_id_to_row,
        codebook_weights_path=data_dir / "rvq_codebook_weights.npy",
    )
    tokenizer.max_items = int(config.get("seq_len", 30))
    tokenizer.config = config
    tokenizer.dataset_dir = str(data_dir)

    requested_device = os.environ.get(
        "GENPLAYLIST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested_device}, but CUDA is unavailable")
    model = Diffusion.load_from_checkpoint(
        str(checkpoint), tokenizer=tokenizer, config=config,
        map_location=requested_device)
    model = model.to(requested_device)
    model.eval()
    _RUNTIME_CACHE = (checkpoint, model, tokenizer, torch)
    return model, tokenizer, torch


def run_backbone(
    context_prefix,
    n_samples,
    catalog_embs,
    catalog_metadata,
    item_id_to_row,
):
    """Generate one next-item slot and decode it to ``GeneratedItem`` objects."""
    model, tokenizer, torch = _load_runtime(
        catalog_metadata, catalog_embs, item_id_to_row)
    context_prefix.validate()
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if len(context_prefix.item_ids) < 2:
        raise ValueError("Next-song inference requires at least two reference music items")
    if len(context_prefix.item_ids) > tokenizer.max_items - 1:
        raise ValueError(
            f"Context has {len(context_prefix.item_ids)} items; maximum is "
            f"{tokenizer.max_items - 1} when reserving one target")

    context_tokens = [tokenizer.bos_token]
    for item_id in context_prefix.item_ids:
        context_tokens.extend(tokenizer.encode_item(item_id))
    context_tokens.append(tokenizer.eos_token)
    device = next(model.parameters()).device
    input_ids = torch.as_tensor(
        context_tokens, dtype=torch.long, device=device)[None, :].repeat(n_samples, 1)

    rows = [item_id_to_row[item_id] for item_id in context_prefix.item_ids]
    context_vectors = np.asarray(catalog_embs[rows], dtype=np.float32)
    mu_c = context_vectors.mean(axis=0, dtype=np.float32)
    sigma_c2 = np.float32(np.mean(np.sum((context_vectors - mu_c) ** 2, axis=1)))
    mu_batch = torch.from_numpy(mu_c)[None, :].repeat(n_samples, 1).to(device)
    sigma_batch = torch.full(
        (n_samples,), float(sigma_c2), dtype=torch.float32, device=device)

    steps = int(model.config.sampling.steps)
    generated = model.restore_model_and_sample_next_item(
        input_ids=input_ids,
        num_steps=steps,
        context_emb=mu_batch,
        mu_c=mu_batch,
        sigma_c2=sigma_batch,
    )
    generated_sequences = generated.detach().cpu().numpy()
    expected_shape = (n_samples, 2 + tokenizer.tokens_per_item)
    if generated_sequences.shape != expected_shape:
        raise ValueError(
            f"Sampler returned shape {generated_sequences.shape}, expected {expected_shape}")

    outputs = []
    for sample_idx, sequence in enumerate(generated_sequences):
        if sequence[0] != tokenizer.bos_token or sequence[-1] != tokenizer.eos_token:
            raise ValueError(f"Sample {sample_idx} lacks BOS/EOS boundaries")
        outputs.append(tokenizer.decode_item(
            sequence[1:-1],
            mu_c=mu_c,
            sigma_c2=float(sigma_c2),
            sample_idx=sample_idx,
            context_prefix=context_prefix,
        ))
    return outputs

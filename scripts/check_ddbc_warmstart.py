#!/usr/bin/env python3
"""Instantiate GenPlaylist and verify a real official DDBC warm start on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WP_ROOT = SRC_ROOT / "03_backbone_recommender"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(WP_ROOT))

from shared.schema import CatalogItem, TOKEN_LAYOUT  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=REPO_ROOT / "checkpoints" / "pretrained" / "ddbc" / "spotify30.ckpt")
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "dataset")
    parser.add_argument(
        "--cue-dir", type=Path,
        default=SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest")
    parser.add_argument(
        "--backward-smoke", action="store_true",
        help="also run one joint-five-target diffusion loss and CPU backward pass")
    args = parser.parse_args()

    import hydra
    import torch
    from omegaconf import OmegaConf

    from diffusion import Diffusion
    from genplaylist_tokenizer import GenPlaylistTokenizer
    from warmstart import apply_ddbc_warmstart

    resolvers = {
        "cwd": lambda: os.getcwd(),
        "device_count": lambda: max(torch.cuda.device_count(), 1),
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(WP_ROOT / "configs")):
        config = hydra.compose(config_name="config", overrides=["model=small"])
    config.data_root = str(args.data_dir.resolve())

    items = CatalogItem.load_catalog(str(args.data_dir / "catalog_metadata.json"))
    mapping = json.loads((args.data_dir / "item_id_to_row.json").read_text(encoding="utf-8"))
    tokenizer = GenPlaylistTokenizer.from_files(
        semantic_tokens_path=args.data_dir / "semantic_tokens.json",
        item2cues_path=args.cue_dir / "item2cues.json",
        cue_manifest_path=args.cue_dir / "cue_manifest.json",
        catalog_items=items,
        catalog_embeddings=np.load(
            args.data_dir / "catalog_item_embeddings.npy", allow_pickle=False),
        item_id_to_row=mapping,
        codebook_weights_path=args.data_dir / "rvq_codebook_weights.npy",
    )
    tokenizer.config = config
    tokenizer.dataset_dir = str(args.data_dir)
    tokenizer.max_items = int(config.seq_len)

    model = Diffusion(config, tokenizer)
    report = apply_ddbc_warmstart(model, args.checkpoint)
    if report["source_runtime_vocab"] != 1028:
        raise ValueError(report)
    if report["target_runtime_vocab"] != TOKEN_LAYOUT.runtime_vocab_size:
        raise ValueError(report)
    if report["new_cue_rows"] != TOKEN_LAYOUT.cue_vocab_size:
        raise ValueError(report)
    print(json.dumps(report, indent=2))
    print("[warmstart] official DDBC weights are compatible via semantic remapping")
    if args.backward_smoke:
        item_ids = [
            item.item_id for item in items[:FROZEN_NEXT_SONG_PROTOCOL.train_total_items]]
        encoded = tokenizer.encode_playlist(
            item_ids, context_items=FROZEN_NEXT_SONG_PROTOCOL.train_reference_items)
        example = {
            "input_ids": encoded.input_ids.tolist(),
            "attention_mask": encoded.attention_mask.tolist(),
            "target_mask": encoded.target_mask.tolist(),
            "sequence_mask": [True] * len(encoded.input_ids),
            "context_emb": encoded.mu_c.tolist(),
            "mu_c": encoded.mu_c.tolist(),
            "sigma_c2": float(encoded.sigma_c2),
        }
        batch = tokenizer.collate_batch([example])
        model.train()
        losses, _ = model._loss(
            batch["input_ids"],
            batch["attention_mask"],
            target_mask=batch["target_mask"],
            context_emb=None,
            mu_c=batch["mu_c"],
            sigma_c2=batch["sigma_c2"],
            sequence_mask=batch["sequence_mask"],
        )
        if not torch.isfinite(losses.loss):
            raise ValueError(f"Non-finite warm-start smoke loss: {losses.loss}")
        losses.loss.backward()
        finite_gradients = sum(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters())
        if finite_gradients == 0:
            raise ValueError("Warm-start backward pass produced no finite gradients")
        print(json.dumps({
            "smoke_loss": float(losses.loss.detach()),
            "target_tokens": int(batch["target_mask"].sum()),
            "sequence_length": int(batch["input_ids"].shape[1]),
            "parameters_with_finite_gradients": finite_gradients,
        }, indent=2))
        print("[warmstart] target-only CPU backward smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Warm-start GenPlaylist from the official DDBC Spotify checkpoint.

The original checkpoint has 1,028 runtime tokens while GenPlaylist has 2,894.
The DiT blocks are shape-compatible, but the token embedding and output head
must be remapped by meaning rather than copied positionally.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.schema import TOKEN_LAYOUT

_VOCAB_STATE_KEYS = (
    "backbone.vocab_embed.embedding",
    "backbone.output_layer.linear.weight",
    "backbone.output_layer.linear.bias",
)


def build_token_row_mapping(
    *,
    source_vocab_size: int,
    source_boi: int,
    source_eos: int,
) -> list[tuple[int, int]]:
    """Return ``(source_row, target_row)`` pairs for shared token meanings."""
    source_mask = source_vocab_size - 1
    if source_boi <= TOKEN_LAYOUT.conflict_token_start:
        raise ValueError(f"Unexpected source BOI token: {source_boi}")
    if source_eos != source_boi + 1 or source_mask != source_eos + 1:
        raise ValueError(
            "Expected legacy special-token order BOI, EOS, MASK at the end of the vocabulary; "
            f"got BOI={source_boi}, EOS={source_eos}, MASK={source_mask}")
    if TOKEN_LAYOUT.mask_token + 1 != TOKEN_LAYOUT.runtime_vocab_size:
        raise ValueError("Target MASK must be the final runtime token")

    # IDs 0..842 retain the same meanings: BOS, 3x256 RVQ entries, and the 74
    # conflict values used by the current catalog. Legacy conflict rows beyond
    # 842 are deliberately not reused as cue embeddings.
    pairs = [(token_id, token_id) for token_id in range(TOKEN_LAYOUT.boi_token)]
    pairs.extend((
        (source_boi, TOKEN_LAYOUT.boi_token),
        (source_eos, TOKEN_LAYOUT.eos_token),
        (source_mask, TOKEN_LAYOUT.mask_token),
    ))
    return pairs


def remap_vocab_rows(source, target, row_mapping: list[tuple[int, int]]):
    """Copy selected vocabulary rows while retaining new cue-row initialization."""
    result = target.clone() if hasattr(target, "clone") else target.copy()
    if source.ndim != target.ndim or source.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"Vocabulary tensor feature shapes differ: source={tuple(source.shape)}, "
            f"target={tuple(target.shape)}")
    for source_row, target_row in row_mapping:
        if not 0 <= source_row < source.shape[0]:
            raise ValueError(f"Source token row {source_row} outside {source.shape[0]}")
        if not 0 <= target_row < target.shape[0]:
            raise ValueError(f"Target token row {target_row} outside {target.shape[0]}")
        result[target_row] = source[source_row]
    return result


def build_warmstart_state(source_checkpoint: dict, target_state: dict):
    """Create a complete target state dict and a human-readable transfer report."""
    try:
        source_state = source_checkpoint["state_dict"]
        source_tokenizer = source_checkpoint["hyper_parameters"]["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Expected a Lightning checkpoint with state_dict and embedded tokenizer") from exc

    source_embedding = source_state.get(_VOCAB_STATE_KEYS[0])
    if source_embedding is None:
        raise ValueError(f"Checkpoint is missing {_VOCAB_STATE_KEYS[0]}")
    row_mapping = build_token_row_mapping(
        source_vocab_size=int(source_embedding.shape[0]),
        source_boi=int(source_tokenizer.boi_token),
        source_eos=int(source_tokenizer.eos_token),
    )

    output = {key: value.clone() for key, value in target_state.items()}
    exact_keys = []
    retained_keys = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if key in _VOCAB_STATE_KEYS:
            if source_value is None:
                raise ValueError(f"Checkpoint is missing vocabulary tensor {key}")
            output[key] = remap_vocab_rows(source_value, target_value, row_mapping)
        elif source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
            output[key] = source_value.clone()
            exact_keys.append(key)
        else:
            retained_keys.append(key)

    report = {
        "source_runtime_vocab": int(source_embedding.shape[0]),
        "target_runtime_vocab": TOKEN_LAYOUT.runtime_vocab_size,
        "mapped_token_rows": len(row_mapping),
        "new_cue_rows": TOKEN_LAYOUT.cue_vocab_size,
        "remapped_keys": list(_VOCAB_STATE_KEYS),
        "exact_keys": exact_keys,
        "retained_target_initialization": retained_keys,
    }
    return output, report


def sync_ema_to_model(model) -> None:
    """Reset EMA shadows after loading warm-start weights into the live model."""
    if model.ema is None:
        return
    parameters = [
        parameter for parameter in chain(model.backbone.parameters(), model.noise.parameters())
        if parameter.requires_grad
    ]
    if len(parameters) != len(model.ema.shadow_params):
        raise ValueError(
            f"EMA parameter count mismatch: {len(parameters)} model vs "
            f"{len(model.ema.shadow_params)} shadow")
    model.ema.shadow_params = [parameter.detach().clone() for parameter in parameters]
    model.ema.collected_params = []
    if model.ema.num_updates is not None:
        model.ema.num_updates = 0


def apply_ddbc_warmstart(model, checkpoint_path: str | Path) -> dict:
    """Load, semantically remap, and apply a trusted official DDBC checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("DDBC warm-start requires PyTorch") from exc
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    remapped, report = build_warmstart_state(checkpoint, model.state_dict())
    model.load_state_dict(remapped, strict=True)
    sync_ema_to_model(model)
    report["checkpoint"] = str(path.resolve())
    report["checkpoint_epoch"] = checkpoint.get("epoch")
    report["checkpoint_global_step"] = checkpoint.get("global_step")
    return report

"""GenPlaylist-v1 tokenizer independent of the legacy DISCO tokenizer.

This is the cross-WP boundary implementation.  It consumes already-built RVQ
semantic IDs and WP-B cue IDs; training the RVQ codebook remains an offline
artifact-building step.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from shared.schema import (  # noqa: E402
    CLHE_EMB_DIM,
    CUE_TOKENS,
    RQ_N_CODEBOOKS,
    CatalogItem,
    GeneratedItem,
    TOKEN_LAYOUT,
)
from shared.artifacts import validate_catalog_alignment  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402


@dataclass(frozen=True)
class TokenizedPlaylist:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    target_mask: np.ndarray
    context_emb: np.ndarray
    mu_c: np.ndarray
    sigma_c2: np.float32


class GenPlaylistTokenizer:
    """Encode the 13-token item representation and decode one candidate."""

    bos_token = 0
    boi_token = TOKEN_LAYOUT.boi_token
    eos_token = TOKEN_LAYOUT.eos_token
    bos_token_id = bos_token
    boi_token_id = boi_token
    eos_token_id = eos_token
    mask_token_id = TOKEN_LAYOUT.mask_token
    vocab_size = TOKEN_LAYOUT.vocab_size
    tokens_per_item = TOKEN_LAYOUT.tokens_per_item

    def __init__(
        self,
        semantic_tokens: dict[str, list[int]],
        item2cues: dict[str, list[int]],
        catalog_items: list[CatalogItem],
        catalog_embeddings: np.ndarray,
        item_id_to_row: dict[str, int],
        codebook_weights: np.ndarray,
    ):
        self.semantic_tokens = {
            str(item_id): [int(token) for token in tokens]
            for item_id, tokens in semantic_tokens.items()
        }
        self.stored_item2cues = {
            str(item_id): [int(cue) for cue in cues]
            for item_id, cues in item2cues.items()
        }
        self.item2cues = {
            item_id: cues[:CUE_TOKENS]
            for item_id, cues in self.stored_item2cues.items()
        }
        self.catalog_items = catalog_items
        self.catalog_embeddings = np.asarray(catalog_embeddings, dtype=np.float32)
        self.item_id_to_row = {str(key): int(value) for key, value in item_id_to_row.items()}
        self.codebook_weights = np.asarray(codebook_weights, dtype=np.float32)
        self.max_items = 30
        self.config = {"rq_codebook_size": TOKEN_LAYOUT.rq_codebook_size}
        self.dataset_dir = None
        self._validate_artifacts()
        self.collate_fn = {
            "train": self.collate_batch,
            "val": self.collate_batch,
            "test": self.collate_batch,
        }

    @classmethod
    def from_dataset_config(cls, config, dataset) -> "GenPlaylistTokenizer":
        """Construct from the canonical repository artifacts named in config."""
        FROZEN_NEXT_SONG_PROTOCOL.validate_config(config)
        data_root = Path(dataset.dir)
        repo_root = Path(__file__).resolve().parents[2]
        cue_root = repo_root / "src" / "02_creative_cues" / "outputs" / "production" / "latest"

        def configured(name: str, default: Path) -> Path:
            value = config.get(name, None)
            return Path(value).expanduser() if value else default

        catalog_items = CatalogItem.load_catalog(str(data_root / "catalog_metadata.json"))
        mapping_path = configured(
            "item_id_to_row_path", data_root / "item_id_to_row.json")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        embeddings_path = configured(
            "catalog_embeddings_path", data_root / "catalog_item_embeddings.npy")
        tokenizer = cls.from_files(
            semantic_tokens_path=configured(
                "semantic_tokens_path", data_root / "semantic_tokens.json"),
            item2cues_path=configured("item2cues_path", cue_root / "item2cues.json"),
            cue_manifest_path=configured("cue_manifest_path", cue_root / "cue_manifest.json"),
            catalog_items=catalog_items,
            catalog_embeddings=np.load(embeddings_path, allow_pickle=False),
            item_id_to_row=mapping,
            codebook_weights_path=configured(
                "codebook_weights_path", data_root / "rvq_codebook_weights.npy"),
            active_cues=int(config.get("active_cue_tokens", CUE_TOKENS)),
        )
        tokenizer.max_items = int(config.get("seq_len", 30))
        tokenizer.config = config
        tokenizer.dataset_dir = str(data_root)
        return tokenizer

    @classmethod
    def from_files(
        cls,
        semantic_tokens_path: str | Path,
        item2cues_path: str | Path,
        cue_manifest_path: str | Path,
        catalog_items: list[CatalogItem],
        catalog_embeddings: np.ndarray,
        item_id_to_row: dict[str, int],
        codebook_weights_path: str | Path,
        active_cues: int = CUE_TOKENS,
    ) -> "GenPlaylistTokenizer":
        manifest = json.loads(Path(cue_manifest_path).read_text(encoding="utf-8"))
        if not manifest.get("wp_d_compatible", False):
            raise ValueError("Cue artifact is marked wp_d_compatible=false")
        if manifest.get("schema_version") != TOKEN_LAYOUT.schema_version:
            raise ValueError(
                f"Cue schema {manifest.get('schema_version')!r} does not match "
                f"{TOKEN_LAYOUT.schema_version!r}")
        if active_cues != CUE_TOKENS:
            raise ValueError(
                f"GenPlaylist-v1 token layout requires active_cues={CUE_TOKENS}, "
                f"got {active_cues}")
        stored_cues = int(manifest.get(
            "stored_cues_per_item", manifest.get("cues_per_item", 0)))
        if stored_cues < active_cues:
            raise ValueError(
                f"Cue artifact stores {stored_cues} cues/item but WP-C needs the "
                f"first {active_cues}")
        manifest_active = int(manifest.get("default_active_cues", CUE_TOKENS))
        if manifest_active != CUE_TOKENS:
            raise ValueError(
                f"Cue artifact default_active_cues={manifest_active} does not match "
                f"the {CUE_TOKENS}-cue token layout")
        semantic_tokens = json.loads(Path(semantic_tokens_path).read_text(encoding="utf-8"))
        item2cues = json.loads(Path(item2cues_path).read_text(encoding="utf-8"))
        bad_lengths = {
            str(item_id): len(cues)
            for item_id, cues in item2cues.items()
            if len(cues) != stored_cues
        }
        if bad_lengths:
            first = next(iter(bad_lengths.items()))
            raise ValueError(
                f"Cue artifact declares {stored_cues} stored cues/item but "
                f"{first[0]} has {first[1]}")
        weights = np.load(codebook_weights_path, allow_pickle=False)
        return cls(
            semantic_tokens, item2cues, catalog_items, catalog_embeddings,
            item_id_to_row, weights)

    def _validate_artifacts(self) -> None:
        validate_catalog_alignment(
            self.catalog_items, self.catalog_embeddings, self.item_id_to_row)
        expected_weights = (
            TOKEN_LAYOUT.rq_n_codebooks * TOKEN_LAYOUT.rq_codebook_size,
            CLHE_EMB_DIM,
        )
        if self.codebook_weights.shape != expected_weights:
            raise ValueError(
                f"RVQ codebook weights must be {expected_weights}, got "
                f"{self.codebook_weights.shape}")
        if not np.isfinite(self.codebook_weights).all():
            raise ValueError("RVQ codebook weights contain NaN or infinity")

        catalog_ids = set(self.item_id_to_row)
        for name, artifact in (
            ("semantic_tokens", self.semantic_tokens), ("item2cues", self.item2cues)):
            missing = sorted(catalog_ids - set(artifact))
            extra = sorted(set(artifact) - catalog_ids)
            if missing or extra:
                raise ValueError(
                    f"{name} ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
        for item_id in catalog_ids:
            self._validated_semantic_tokens(item_id)
            stored_cues = self.stored_item2cues[item_id]
            if len(stored_cues) < CUE_TOKENS:
                raise ValueError(
                    f"Item {item_id} stores only {len(stored_cues)} cues; "
                    f"at least {CUE_TOKENS} are required")
            cues = self.item2cues[item_id]
            if len(cues) != CUE_TOKENS or any(
                cue < 0 or cue >= TOKEN_LAYOUT.cue_vocab_size for cue in cues):
                raise ValueError(f"Invalid cue IDs for item {item_id}: {cues}")

    def _validated_semantic_tokens(self, item_id: str) -> list[int]:
        tokens = self.semantic_tokens[item_id]
        if len(tokens) != RQ_N_CODEBOOKS + 1:
            raise ValueError(f"Item {item_id} needs 3 RVQ tokens + 1 conflict token")
        for level, token in enumerate(tokens[:RQ_N_CODEBOOKS]):
            lower = TOKEN_LAYOUT.rvq_token(level, 0)
            upper = TOKEN_LAYOUT.rvq_token(level, TOKEN_LAYOUT.rq_codebook_size - 1)
            if not lower <= token <= upper:
                raise ValueError(
                    f"Item {item_id} RVQ level {level} token {token} outside {lower}..{upper}")
        conflict = tokens[-1]
        lower = TOKEN_LAYOUT.conflict_token(0)
        upper = TOKEN_LAYOUT.conflict_token(TOKEN_LAYOUT.conflict_vocab_size - 1)
        if not lower <= conflict <= upper:
            raise ValueError(
                f"Item {item_id} conflict token {conflict} outside {lower}..{upper}")
        return tokens

    def encode_item(self, item_id: str) -> list[int]:
        item_id = str(item_id)
        if item_id not in self.semantic_tokens:
            raise KeyError(f"Unknown item ID: {item_id}")
        semantic = self._validated_semantic_tokens(item_id)
        cues = [TOKEN_LAYOUT.cue_token(cue) for cue in self.item2cues[item_id]]
        return [self.boi_token, *semantic, *cues]

    def encode_playlist(self, item_ids: list[str], context_items: int) -> TokenizedPlaylist:
        ids = [str(item_id) for item_id in item_ids]
        if len(ids) < 3:
            raise ValueError(
                "Next-song training needs at least two reference items and one target item")
        if not 2 <= context_items < len(ids):
            raise ValueError(
                "context_items must contain at least two references and leave a target item")
        if len(set(ids)) != len(ids):
            raise ValueError("Playlist item IDs must be unique")

        sequence = [self.bos_token]
        target_mask = [False]
        for item_index, item_id in enumerate(ids):
            encoded = self.encode_item(item_id)
            sequence.extend(encoded)
            is_target = item_index >= context_items
            target_mask.extend([False] + [is_target] * (self.tokens_per_item - 1))
        sequence.append(self.eos_token)
        target_mask.append(False)

        input_ids = np.asarray(sequence, dtype=np.int64)
        special = np.isin(input_ids, [self.bos_token, self.boi_token, self.eos_token])
        attention_mask = ~special
        context_rows = [self.item_id_to_row[item_id] for item_id in ids[:context_items]]
        context_emb = self.catalog_embeddings[context_rows]
        mu_c = context_emb.mean(axis=0, dtype=np.float32)
        sigma_c2 = np.float32(np.mean(np.sum((context_emb - mu_c) ** 2, axis=1)))
        return TokenizedPlaylist(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=np.asarray(target_mask, dtype=bool),
            context_emb=context_emb,
            mu_c=mu_c,
            sigma_c2=sigma_c2,
        )

    def build_next_item_completion(
        self, context_tokens: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``references + [BOI, MASK x 12, EOS]`` for full-mask inference."""
        values = np.asarray(context_tokens, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(f"context_tokens must be 1-D, got {values.shape}")
        if len(values) < 2 or values[0] != self.bos_token or values[-1] != self.eos_token:
            raise ValueError("Context must be bounded by exactly one leading BOS and trailing EOS")
        if np.count_nonzero(values == self.eos_token) != 1:
            raise ValueError("Context must contain exactly one EOS")
        if np.any(values == self.mask_token_id):
            raise ValueError("Reference context must not contain MASK tokens")
        reference_width = len(values) - 2
        if reference_width % self.tokens_per_item != 0:
            raise ValueError("Reference context contains a partial item")
        if reference_width // self.tokens_per_item < 2:
            raise ValueError("Next-song completion requires at least two reference items")

        payload_width = self.tokens_per_item - 1
        completed = np.asarray([
            *values[:-1],
            self.boi_token,
            *([self.mask_token_id] * payload_width),
            self.eos_token,
        ], dtype=np.int64)
        completion_mask = np.zeros(len(completed), dtype=bool)
        payload_start = len(values)
        completion_mask[payload_start:payload_start + payload_width] = True
        return completed, completion_mask

    def make_type_mask(self, seq_len: int) -> np.ndarray:
        """Return ``[seq_len, runtime_vocab]`` legal-token positions.

        The final position is reserved for EOS; every complete item payload
        between BOS/EOS follows the 13-token stride. MASK is never a
        legal clean prediction.
        """
        if seq_len < 2 or (seq_len - 2) % self.tokens_per_item != 0:
            raise ValueError(
                f"Sequence length must be 2 + n*{self.tokens_per_item}, got {seq_len}")
        legal = np.zeros((seq_len, TOKEN_LAYOUT.runtime_vocab_size), dtype=bool)
        legal[0, self.bos_token] = True
        legal[-1, self.eos_token] = True
        for position in range(1, seq_len - 1):
            offset = (position - 1) % self.tokens_per_item
            if offset == 0:
                legal[position, self.boi_token] = True
            elif 1 <= offset <= RQ_N_CODEBOOKS:
                level = offset - 1
                start = TOKEN_LAYOUT.rvq_token(level, 0)
                legal[position, start:start + TOKEN_LAYOUT.rq_codebook_size] = True
            elif offset == RQ_N_CODEBOOKS + 1:
                start = TOKEN_LAYOUT.conflict_token(0)
                legal[position, start:start + TOKEN_LAYOUT.conflict_vocab_size] = True
            else:
                start = TOKEN_LAYOUT.cue_token(0)
                legal[position, start:start + TOKEN_LAYOUT.cue_vocab_size] = True
        return legal

    def decode_item(
        self,
        tokens: list[int] | np.ndarray,
        *,
        mu_c: np.ndarray,
        sigma_c2: float,
        sample_idx: int = 0,
        context_prefix=None,
    ) -> GeneratedItem:
        values = [int(token) for token in tokens]
        if values and values[0] == self.boi_token:
            values = values[1:]
        if len(values) != self.tokens_per_item - 1:
            raise ValueError(
                f"Expected {self.tokens_per_item - 1} item payload tokens, got {len(values)}")
        semantic = values[:RQ_N_CODEBOOKS + 1]
        cue_tokens = values[RQ_N_CODEBOOKS + 1:]
        rvq_codes = tuple(
            semantic[level] - TOKEN_LAYOUT.rvq_token(level, 0)
            for level in range(RQ_N_CODEBOOKS)
        )
        conflict_code = semantic[-1] - TOKEN_LAYOUT.conflict_token(0)
        cue_ids = [token - TOKEN_LAYOUT.cue_token(0) for token in cue_tokens]
        # Reuse the same range checks used for catalog semantic tokens.
        synthetic_id = "<generated>"
        old = self.semantic_tokens.get(synthetic_id)
        self.semantic_tokens[synthetic_id] = semantic
        try:
            self._validated_semantic_tokens(synthetic_id)
        finally:
            if old is None:
                del self.semantic_tokens[synthetic_id]
            else:
                self.semantic_tokens[synthetic_id] = old
        if any(cue < 0 or cue >= TOKEN_LAYOUT.cue_vocab_size for cue in cue_ids):
            raise ValueError(f"Generated cue token outside cue range: {cue_tokens}")

        rows = [level * TOKEN_LAYOUT.rq_codebook_size + code
                for level, code in enumerate(rvq_codes)]
        z_hat = self.codebook_weights[rows].sum(axis=0).astype(np.float32)
        generated = GeneratedItem(
            rvq_codes=rvq_codes,
            conflict_code=conflict_code,
            z_hat_emb=z_hat,
            mu_c_emb=np.asarray(mu_c, dtype=np.float32),
            sigma_c2=float(sigma_c2),
            cue_ids=cue_ids,
            sample_idx=sample_idx,
            context_prefix=context_prefix,
        )
        return generated.validate()

    def _token_to_feature(self, semantic_tokens) -> np.ndarray:
        """Compatibility decoder for evaluator paths (3 RVQ + conflict)."""
        values = [int(token) for token in semantic_tokens]
        if len(values) != RQ_N_CODEBOOKS + 1:
            raise ValueError(f"Expected four semantic tokens, got {len(values)}")
        rvq_codes = [
            values[level] - TOKEN_LAYOUT.rvq_token(level, 0)
            for level in range(RQ_N_CODEBOOKS)
        ]
        for level, code in enumerate(rvq_codes):
            TOKEN_LAYOUT.rvq_token(level, code)
        TOKEN_LAYOUT.conflict_token(values[-1] - TOKEN_LAYOUT.conflict_token(0))
        rows = [level * TOKEN_LAYOUT.rq_codebook_size + code
                for level, code in enumerate(rvq_codes)]
        return self.codebook_weights[rows].sum(axis=0).astype(np.float32)

    @property
    def n_digit(self) -> int:
        return RQ_N_CODEBOOKS

    @property
    def padding_token(self) -> int:
        return self.bos_token

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.max_items * self.tokens_per_item + 1

    def tokenize(self, datasets: dict) -> dict:
        """Tokenize next-one training rows and fixed 15->5 evaluation rows."""
        tokenized = {}
        for split, source in datasets.items():
            usable = source.filter(lambda row: len(row["item_seq"]) >= 3)

            def encode_row(row):
                item_ids = [str(item_id) for item_id in row["item_seq"]]
                if split == "test":
                    protocol = FROZEN_NEXT_SONG_PROTOCOL.validate_config(self.config)
                    reference_count = protocol.eval_reference_items
                    target_count = protocol.eval_target_items
                    expected_count = protocol.eval_total_items
                    if len(item_ids) != expected_count:
                        raise ValueError(
                            f"Test rows must contain exactly {expected_count} items for "
                            f"{reference_count}->{target_count} evaluation, got {len(item_ids)}")
                    reference_ids = item_ids[:reference_count]
                    target_ids = item_ids[reference_count:]
                    # Reuse encode_playlist to compute the context statistics;
                    # the temporary target is not exposed in test input_ids.
                    encoded = self.encode_playlist(
                        [*reference_ids, target_ids[0]], context_items=reference_count)
                else:
                    reference_ids = item_ids[:-1]
                    target_ids = item_ids[-1:]
                    encoded = self.encode_playlist(
                        item_ids, context_items=len(reference_ids))
                result = {
                    "input_ids": encoded.input_ids.tolist(),
                    "sequence_mask": [True] * len(encoded.input_ids),
                    "attention_mask": encoded.attention_mask.tolist(),
                    "target_mask": encoded.target_mask.tolist(),
                    # Mean context is the portable CFG path; mu_c is also fed
                    # independently into AdaLN structure conditioning.
                    "context_emb": encoded.mu_c.tolist(),
                    "mu_c": encoded.mu_c.tolist(),
                    "sigma_c2": float(encoded.sigma_c2),
                }
                if split == "test":
                    context_tokens = [self.bos_token]
                    for item_id in reference_ids:
                        context_tokens.extend(self.encode_item(item_id))
                    context_tokens.append(self.eos_token)
                    result["input_ids"] = context_tokens
                    result["sequence_mask"] = [True] * len(context_tokens)
                    result["attention_mask"] = [
                        token not in (self.bos_token, self.boi_token, self.eos_token)
                        for token in context_tokens]
                    result["target_mask"] = [False] * len(context_tokens)
                    result["labels"] = [
                        self._validated_semantic_tokens(item_id)
                        for item_id in target_ids]
                return result

            tokenized[split] = usable.map(
                encode_row,
                remove_columns=usable.column_names,
                desc=f"Tokenizing {split} set (GenPlaylist v1)",
            )
            tokenized[split].set_format(type="torch")
        return tokenized

    def collate_batch(self, examples: list[dict]):
        """Pad variable playlist lengths while keeping all padding/context fixed."""
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(example["input_ids"]) for example in examples)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_length), self.padding_token, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        sequence_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        for row, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[row, :length] = torch.as_tensor(example["input_ids"], dtype=torch.long)
            sequence_mask[row, :length] = True
            attention_mask[row, :length] = torch.as_tensor(
                example["attention_mask"], dtype=torch.bool)
            target_mask[row, :length] = torch.as_tensor(
                example["target_mask"], dtype=torch.bool)
        batch = {
            "input_ids": input_ids,
            "sequence_mask": sequence_mask,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "context_emb": torch.stack([
                torch.as_tensor(example["context_emb"], dtype=torch.float32)
                for example in examples]),
            "mu_c": torch.stack([
                torch.as_tensor(example["mu_c"], dtype=torch.float32)
                for example in examples]),
            "sigma_c2": torch.as_tensor(
                [example["sigma_c2"] for example in examples], dtype=torch.float32),
        }
        if "labels" in examples[0]:
            batch["labels"] = torch.stack([
                torch.as_tensor(example["labels"], dtype=torch.long)
                for example in examples])
        return batch

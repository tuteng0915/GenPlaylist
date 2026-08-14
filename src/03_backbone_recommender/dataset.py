"""Dataset adapter for the frozen GenPlaylist catalog and playlist splits.

The source files use opaque, sparse item IDs.  This module deliberately keeps
them as strings; conversion to embedding rows happens only through the shared
``item_id_to_row.json`` artifact.
"""

from __future__ import annotations

import json
import sys
from logging import getLogger
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402

try:
    from datasets import Dataset
except ImportError:  # lets lightweight schema/parser checks run without WP-C extras
    Dataset = None


def read_split_file(path: str | Path) -> list[tuple[str, list[str]]]:
    """Read ``playlist_id, item_id, ...`` records without coercing IDs to ints."""
    records: list[tuple[str, list[str]]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            fields = [field.strip() for field in raw.strip().split(",")]
            if not fields or fields == [""]:
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected playlist ID and at least two item IDs")
            playlist_id, item_ids = fields[0], fields[1:]
            if not playlist_id or any(not item_id for item_id in item_ids):
                raise ValueError(f"{path}:{line_no}: empty playlist/item ID")
            records.append((playlist_id, item_ids))
    return records


class AbstractDataset:
    """Compatibility wrapper consumed by the WP-C training entry point."""

    def __init__(self, config: dict):
        if Dataset is None:
            raise RuntimeError(
                "WP-C requires the 'datasets' package; install requirements.txt")
        self.config = config
        self.logger = getLogger()
        repo_root = Path(__file__).resolve().parents[2]
        configured_root = config.get("data_root", None)
        self.dir = str(
            Path(configured_root).expanduser().resolve()
            if configured_root
            else repo_root / "data" / "dataset"
        )
        self._root = Path(self.dir)
        self._require_files()

        with (self._root / "catalog_metadata.json").open("r", encoding="utf-8") as f:
            catalog = json.load(f)
        if not isinstance(catalog, dict):
            raise ValueError("catalog_metadata.json must be keyed by opaque item ID")
        self.item2meta = {str(item_id): meta for item_id, meta in catalog.items()}

        card_path = self._root / "dataset_card.json"
        self.dataset_card = (
            json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        )
        self._split_cache: dict[str, Dataset] | None = None
        self.split_data = self.split()
        self.bi_full = [
            item_ids
            for split_name in ("train", "test")
            for _, item_ids in self._records_for_split(split_name)
        ]

    def _require_files(self) -> None:
        required = [
            self._root / "catalog_metadata.json",
            self._root / "splits" / "train.txt",
            self._root / "splits" / "val.txt",
            self._root / "splits" / "test.txt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "GenPlaylist dataset is incomplete; missing: " + ", ".join(missing))

    def _records_for_split(self, split: str) -> list[tuple[str, list[str]]]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split {split!r}; GenPlaylist-v4 exposes train/test only")
        source_names = ("val", "test") if split == "test" else (split,)
        records = [
            record
            for source_name in source_names
            for record in read_split_file(self._root / "splits" / f"{source_name}.txt")
        ]
        known_ids = set(self.item2meta)
        unknown = sorted({item_id for _, seq in records for item_id in seq} - known_ids)
        if unknown:
            raise ValueError(
                f"Split sources {source_names!r} reference {len(unknown)} "
                "catalog-missing IDs: "
                f"{unknown[:5]}")
        playlist_ids = [playlist_id for playlist_id, _ in records]
        if len(set(playlist_ids)) != len(playlist_ids):
            raise ValueError(f"Split sources {source_names!r} contain duplicate playlist IDs")
        return records

    def convert_txt_to_dataset(
        self,
        file_name: str,
        swap_ratio: float,
        seq_len: int,
        if_train: bool = False,
    ) -> dict[str, list]:
        """Build frozen joint 15-reference/5-target train and test examples.

        Training expands every eligible playlist into all chronological
        20-item rolling windows (stride one).  Every window contains exactly
        15 references followed by the next five targets.

        Test rows combine the original val/test sources and follow the fixed
        15->5 protocol: playlists shorter than 20 items are excluded and the
        first 20 chronological items are retained.
        The tokenizer later exposes the first 15 as context and the last five
        as the unordered ground-truth set for many-to-many evaluation.
        """
        records = self._records_for_split(file_name)
        output: list[tuple[str, list[str]]] = []
        protocol = FROZEN_NEXT_SONG_PROTOCOL.validate_config(self.config)
        reference_items = protocol.train_reference_items
        target_items = protocol.train_target_items
        if seq_len != protocol.train_total_items:
            raise ValueError(
                f"seq_len={seq_len} must equal frozen train_total_items="
                f"{protocol.train_total_items}")
        if if_train and swap_ratio:
            raise ValueError(
                "swap_ratio must be 0 for chronological continuation training")

        for playlist_id, item_ids in records:
            if if_train:
                for window_start in range(0, len(item_ids) - seq_len + 1):
                    example = item_ids[window_start:window_start + seq_len]
                    if len(example[:reference_items]) != reference_items or len(
                            example[reference_items:]) != target_items:
                        raise AssertionError("Malformed joint 15->5 training window")
                    output.append((f"{playlist_id}:joint5:{window_start}", example))
            elif file_name == "test":
                if len(item_ids) >= protocol.eval_total_items:
                    references, targets = protocol.split_evaluation_items(item_ids)
                    output.append((playlist_id, [*references, *targets]))
        return {
            "bundle": [playlist_id for playlist_id, _ in output],
            "item_seq": [item_ids for _, item_ids in output],
        }

    def split(self) -> dict[str, Dataset]:
        if self._split_cache is None:
            swap_ratio = float(self.config.get("swap_ratio", 0.0))
            seq_len = int(self.config.get("seq_len", 0))
            self._split_cache = {
                split: Dataset.from_dict(self.convert_txt_to_dataset(
                    split, swap_ratio, seq_len, if_train=(split == "train")))
                for split in ("train", "test")
            }
        return self._split_cache

    def __str__(self) -> str:
        return (
            f"[Dataset] {self.dir}\n"
            f"\tNumber of playlists: {self.n_bundle}\n"
            f"\tNumber of items: {self.n_items}\n"
            f"\tPlaylist-item interactions: {self.bi_interactions}\n"
            f"\tMax items / playlist: {self.max_item_seq_len}\n"
        )

    @property
    def max_item_seq_len(self) -> int:
        lengths = [len(seq) for seq in self.bi_full]
        return max(lengths, default=0)

    @property
    def n_bundle(self) -> int:
        return len(self.bi_full)

    @property
    def n_users(self) -> int:
        return 0

    @property
    def n_items(self) -> int:
        return len(self.item2meta)

    @property
    def ui_interactions(self) -> int:
        return 0

    @property
    def bi_interactions(self) -> int:
        return sum(len(seq) for seq in self.bi_full)

    @property
    def n_interactions(self) -> int:
        return self.bi_interactions

    @property
    def avg_item_seq_len(self) -> float:
        return self.bi_interactions / max(self.n_bundle, 1)

    def log(self, message, level="info"):
        from utils import log
        return log(message, self.logger, level=level)

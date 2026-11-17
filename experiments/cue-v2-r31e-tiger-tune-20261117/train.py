"""Music4All TIGER+cues warm-start trial, isolated from frozen v2 results.

This finite training-only study starts from the corrected SID-only checkpoint,
adds v2 catalog cues to the encoder and a cue head to the decoder, and varies
only the auxiliary cue-loss weight.  It never writes to a baseline namespace
and does not read test examples while training.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import Adafactor


R = Path("/storage_fast/ttu/GenPlaylist")
E = R / "experiments/cue-v2-r31e-tiger-tune-20261117"
O = R / "outputs/cue-v2-r31e-tiger-tune-20261117"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save(path: Path, value) -> None:
    temporary = path.with_suffix(".new")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue-weight", choices=("0.02", "0.05"), required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--bootstrap-steps", type=int, default=200)
    args = parser.parse_args()
    assert args.steps > args.bootstrap_steps >= 0

    arm = "warm-ramp-" + args.cue_weight.replace(".", "p")
    out = O / "m4a" / arm
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    torch.set_num_threads(4)
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    core = load_module(
        "tiger_v2_joint_core",
        R / "experiments/cue-v2-r31e-tiger-20261113/core.py",
    )
    prepared = R / "original-ncl/data/processed/genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16"
    split_path = R / "experiments/two-dataset-fair-20261108/split.json"
    split = json.loads(split_path.read_text())
    assert sha(split_path) == "2d4443d88d2e153a6e0b6d1fc72d76568cd979d1f52fd0d9a161bd9deff8f2a7"
    assert sha(prepared / "prepared_manifest.json") == split["source_manifest_sha256"]

    vectors = prepared / "vectors"
    ids = list(map(str, json.loads((vectors / "catalog_item_ids.json").read_text())))
    rows = {item: index for index, item in enumerate(ids)}
    canonical = np.load(vectors / "catalog_semantic_tokens.npy").astype(np.int64)
    cue_root = R / "outputs/cue-v2-r31e-20261113"
    item_cues = json.loads((cue_root / "m4a/item2cues.json").read_text())
    cues = np.asarray([item_cues[item][:8] for item in ids], dtype=np.int64)
    cat = core.Catalog(canonical, cues, len(json.loads((cue_root / "cue_vocab.json").read_text())))

    base_checkpoint = R / "outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/m4a/sid/last.ckpt"
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    assert base["step"] == 20000
    model = core.build_model(cat.vocab_size)
    model.load_state_dict(base["model"])
    old_vocab = int(model.config.vocab_size)
    assert old_vocab == cat.vocab_size
    del base
    model = model.to(device)
    parameters = list(model.parameters())
    optimizer = Adafactor(
        parameters,
        lr=1e-4,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.0,
    )

    raw = load_from_disk(str(prepared / "raw_dataset"))
    train_indices = split["training_indices"]
    validation_indices = split["validation_indices"][:128]
    assert not set(train_indices) & set(validation_indices)
    training = torch.tensor(
        [[rows[str(item)] for item in row] for row in raw["train"].select(train_indices)["item_seq"]]
    )
    validation = torch.tensor(
        [[rows[str(item)] for item in row] for row in raw["train"].select(validation_indices)["item_seq"]]
    )

    identity = {
        "schema": "tiger-v2-warm-start-cue-ramp-v1",
        "dataset": "m4a",
        "arm": arm,
        "steps": args.steps,
        "bootstrap_steps": args.bootstrap_steps,
        "cue_weight": float(args.cue_weight),
        "seed": 101,
        "sampling_seed": 20261117,
        "source_sid_checkpoint": str(base_checkpoint),
        "source_sid_checkpoint_sha256": sha(base_checkpoint),
        "v2_item_cues_sha256": sha(cue_root / "m4a/item2cues.json"),
        "v2_cue_vocab_sha256": sha(cue_root / "cue_vocab.json"),
        "split_sha256": sha(split_path),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "optimizer": "Adafactor(lr=1e-4, no relative/scale schedule)",
        "objective": "SID CE + ramped cue CE + 0.1 EOS CE",
        "encoder": "15..19 history items, each SID4+v2 catalog cue8; no future target",
        "decoder": "SID4 then cue8 then EOS",
        "test_selection": "none; fixed training-source validation only",
        "code_sha256": sha(Path(__file__)),
    }
    identity_path = out / "identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        dump(identity_path, identity)

    checkpoint = out / "last.ckpt"
    generator = torch.Generator(device="cpu").manual_seed(20261117)
    start = 0
    digest = "00" * 32
    elapsed_before = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["identity"] == identity
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        start, digest, elapsed_before = saved["step"], saved["digest"], saved["seconds"]
        del saved

    should_stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: should_stop.__setitem__(0, True))
    started = time.monotonic()

    def checkpoint_now(step: int) -> None:
        save(
            checkpoint,
            {
                "identity": identity,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampling_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "digest": digest,
                "seconds": elapsed_before + time.monotonic() - started,
            },
        )
        dump(
            out / "progress.json",
            {"step": step, "steps": args.steps, "seconds": elapsed_before + time.monotonic() - started, "digest": digest},
        )

    @torch.inference_mode()
    def validation_losses() -> dict:
        model.eval()
        values = {"sid": [], "cue": [], "eos": []}
        validation_generator = torch.Generator(device="cpu").manual_seed(991)
        for _ in range(8):
            x, mask, y, _, _ = cat.batch(validation, 16, validation_generator, True, device)
            logits = model(input_ids=x, attention_mask=mask, labels=y).logits
            ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
            values["sid"].append(float(ce[:, :4].mean()))
            values["cue"].append(float(ce[:, 4:12].mean()))
            values["eos"].append(float(ce[:, -1].mean()))
        model.train()
        return {name: float(np.mean(value)) for name, value in values.items()}

    model.train()
    for step in range(start + 1, args.steps + 1):
        x, mask, y, selected_rows, selected_slots = cat.batch(training, 256, generator, True, device)
        digest = hashlib.sha256(
            bytes.fromhex(digest) + selected_rows.numpy().tobytes() + selected_slots.numpy().tobytes()
        ).hexdigest()
        cue_weight = float(args.cue_weight) * min(1.0, max(0.0, (step - args.bootstrap_steps) / 800.0))
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=x, attention_mask=mask, labels=y).logits
        ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
        sid = ce[:, :4].mean()
        cue = ce[:, 4:12].mean()
        eos = ce[:, -1].mean()
        primary = sid + 0.1 * eos
        auxiliary = cue_weight * cue
        if step <= args.bootstrap_steps:
            # Bootstrap only the cue embedding/output rows; preserve the SID model.
            (0.1 * cue).backward()
            for parameter in parameters:
                if parameter is model.shared.weight:
                    parameter.grad[:cat.cue_offset].zero_()
                else:
                    parameter.grad = None
        else:
            (primary + auxiliary).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % 25 == 0:
            row = {
                "step": step,
                "sid_loss": float(sid),
                "cue_loss": float(cue),
                "eos_loss": float(eos),
                "cue_weight": cue_weight,
                "bootstrap": step <= args.bootstrap_steps,
                "grad_norm": float(norm),
                "seconds": elapsed_before + time.monotonic() - started,
            }
            with (out / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % 250 == 0 or step == args.steps or should_stop[0]:
            checkpoint_now(step)
        if step % 500 == 0:
            report = {"step": step, "losses": validation_losses()}
            dump(out / f"validation-step{step}.json", report)
            print(json.dumps({"validation": report}), flush=True)
        if should_stop[0]:
            return
    dump(out / "complete.json", {"steps": args.steps, "checkpoint_sha256": sha(checkpoint), "identity_sha256": sha(identity_path)})


if __name__ == "__main__":
    main()

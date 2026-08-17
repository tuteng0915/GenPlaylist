#!/usr/bin/env python3
"""Serve the frozen blinded listener study without modifying the WP-D demo."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402


RATINGS = {1, 2, 3, 4, 5}
PREFERENCES = {"Song A", "Song B", "No preference"}
LISTENING_FREQUENCIES = {
    "Daily", "Several times a week", "Weekly", "Occasionally", "Rarely"}
MUSICAL_TRAINING = {"Yes", "No"}


class AlreadySubmittedError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS assignments (
            participant_hash TEXT PRIMARY KEY,
            session_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            display_song_a_is_generated INTEGER NOT NULL CHECK (
                display_song_a_is_generated IN (0, 1)),
            consented_utc TEXT NOT NULL,
            assigned_utc TEXT NOT NULL,
            submitted_utc TEXT
        );
        CREATE TABLE IF NOT EXISTS responses (
            session_id TEXT PRIMARY KEY,
            participant_hash TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            song_a_is_generated INTEGER NOT NULL CHECK (song_a_is_generated IN (0, 1)),
            fit_a INTEGER NOT NULL CHECK (fit_a BETWEEN 1 AND 5),
            fit_b INTEGER NOT NULL CHECK (fit_b BETWEEN 1 AND 5),
            quality_a INTEGER NOT NULL CHECK (quality_a BETWEEN 1 AND 5),
            quality_b INTEGER NOT NULL CHECK (quality_b BETWEEN 1 AND 5),
            novelty_a INTEGER NOT NULL CHECK (novelty_a BETWEEN 1 AND 5),
            novelty_b INTEGER NOT NULL CHECK (novelty_b BETWEEN 1 AND 5),
            preference TEXT NOT NULL,
            listening_freq TEXT NOT NULL,
            musical_training TEXT NOT NULL,
            playback_confirmed INTEGER NOT NULL CHECK (playback_confirmed = 1),
            notes TEXT NOT NULL,
            submitted_utc TEXT NOT NULL,
            FOREIGN KEY (participant_hash) REFERENCES assignments(participant_hash)
        );
        """
    )


def _participant_hash(code: str, key: bytes) -> str:
    normalized = code.strip()
    if not 3 <= len(normalized) <= 128:
        raise ValueError("Participant code must contain 3 to 128 characters")
    return hmac.new(key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def _choose_balanced_case(
    connection: sqlite3.Connection, case_ids: tuple[str, ...], chooser,
) -> str:
    counts = {case_id: 0 for case_id in case_ids}
    for row in connection.execute(
        "SELECT case_id, COUNT(*) AS count FROM assignments GROUP BY case_id"
    ):
        if row["case_id"] not in counts:
            raise ValueError(f"Database contains an unknown case: {row['case_id']}")
        counts[row["case_id"]] = int(row["count"])
    minimum = min(counts.values())
    return str(chooser([case_id for case_id, count in counts.items() if count == minimum]))


def _choose_balanced_side(
    connection: sqlite3.Connection, case_id: str, chooser,
) -> bool:
    counts = {True: 0, False: 0}
    for row in connection.execute(
        """SELECT display_song_a_is_generated, COUNT(*) AS count
           FROM assignments WHERE case_id = ?
           GROUP BY display_song_a_is_generated""",
        (case_id,),
    ):
        counts[bool(row["display_song_a_is_generated"])] = int(row["count"])
    if counts[True] < counts[False]:
        return True
    if counts[False] < counts[True]:
        return False
    return bool(chooser([True, False]))


def _assign_participant(
    connection: sqlite3.Connection,
    participant_hash: str,
    case_ids: tuple[str, ...],
    chooser=secrets.choice,
) -> dict:
    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = connection.execute(
            "SELECT * FROM assignments WHERE participant_hash = ?",
            (participant_hash,),
        ).fetchone()
        if existing is not None:
            if existing["submitted_utc"] is not None:
                raise AlreadySubmittedError("This participant code has already submitted")
            connection.execute("COMMIT")
            return dict(existing)
        case_id = _choose_balanced_case(connection, case_ids, chooser)
        generated_is_a = _choose_balanced_side(connection, case_id, chooser)
        now = _utc_now()
        assignment = {
            "participant_hash": participant_hash,
            "session_id": uuid.uuid4().hex,
            "case_id": case_id,
            "display_song_a_is_generated": int(generated_is_a),
            "consented_utc": now,
            "assigned_utc": now,
            "submitted_utc": None,
        }
        connection.execute(
            """INSERT INTO assignments
               (participant_hash, session_id, case_id,
                display_song_a_is_generated, consented_utc, assigned_utc,
                submitted_utc)
               VALUES (:participant_hash, :session_id, :case_id,
                       :display_song_a_is_generated, :consented_utc,
                       :assigned_utc, :submitted_utc)""",
            assignment,
        )
        connection.execute("COMMIT")
        return assignment
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _validated_rating(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"Please provide {name}")
    numeric = int(value)
    if numeric not in RATINGS:
        raise ValueError(f"{name} must be an integer from 1 to 5")
    return numeric


def _submit_response(
    connection: sqlite3.Connection,
    session_id: str,
    ratings: dict[str, object],
    preference: str,
    listening_freq: str,
    musical_training: str,
    playback_confirmed: bool,
    notes: str,
) -> None:
    if preference not in PREFERENCES:
        raise ValueError("Please provide an overall preference")
    if listening_freq not in LISTENING_FREQUENCIES:
        raise ValueError("Please provide listening frequency")
    if musical_training not in MUSICAL_TRAINING:
        raise ValueError("Please provide musical-training status")
    if not playback_confirmed:
        raise ValueError("Please confirm that both candidates were played")
    notes = str(notes or "").strip()
    if len(notes) > 2000:
        raise ValueError("Comments must contain at most 2,000 characters")
    cleaned = {
        name: _validated_rating(ratings.get(name), name)
        for name in (
            "fit_a", "fit_b", "quality_a", "quality_b", "novelty_a", "novelty_b")
    }

    connection.execute("BEGIN IMMEDIATE")
    try:
        assignment = connection.execute(
            "SELECT * FROM assignments WHERE session_id = ?", (session_id,)
        ).fetchone()
        if assignment is None:
            raise ValueError("Unknown or expired study session")
        if assignment["submitted_utc"] is not None:
            raise AlreadySubmittedError("This study session has already submitted")
        now = _utc_now()
        connection.execute(
            """INSERT INTO responses
               (session_id, participant_hash, case_id, song_a_is_generated,
                fit_a, fit_b, quality_a, quality_b, novelty_a, novelty_b,
                preference, listening_freq, musical_training,
                playback_confirmed, notes, submitted_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                session_id,
                assignment["participant_hash"],
                assignment["case_id"],
                assignment["display_song_a_is_generated"],
                cleaned["fit_a"], cleaned["fit_b"],
                cleaned["quality_a"], cleaned["quality_b"],
                cleaned["novelty_a"], cleaned["novelty_b"],
                preference, listening_freq, musical_training, notes, now,
            ),
        )
        connection.execute(
            "UPDATE assignments SET submitted_utc = ? WHERE session_id = ?",
            (now, session_id),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_cases(study_dir: Path) -> dict[str, dict]:
    public_path = study_dir / "public_manifest.json"
    private_path = study_dir / "private_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private["inputs"]["public_manifest_sha256"] != sha256_file(public_path):
        raise ValueError("Public study manifest hash mismatch")
    public_by_id = {case["case_id"]: case for case in public["cases"]}
    private_by_id = {case["case_id"]: case for case in private["cases"]}
    if set(public_by_id) != set(private_by_id) or len(public_by_id) != 25:
        raise ValueError("Public/private study cases are incomplete or misaligned")
    cases = {}
    for case_id, public_case in public_by_id.items():
        private_case = private_by_id[case_id]
        path_a = (study_dir / public_case["song_a"]).resolve()
        path_b = (study_dir / public_case["song_b"]).resolve()
        assets_root = (study_dir / "participant_assets").resolve()
        if assets_root not in path_a.parents or assets_root not in path_b.parents:
            raise ValueError(f"Study asset escapes participant directory: {case_id}")
        if sha256_file(path_a) != private_case["song_a_sha256"]:
            raise ValueError(f"Song A hash mismatch: {case_id}")
        if sha256_file(path_b) != private_case["song_b_sha256"]:
            raise ValueError(f"Song B hash mismatch: {case_id}")
        cases[case_id] = {
            **public_case,
            "path_a": path_a,
            "path_b": path_b,
            "base_song_a_is_generated": bool(private_case["generated_is_song_a"]),
        }
    return cases


def _load_or_create_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secrets.token_bytes(32))
    path.chmod(0o600)
    value = path.read_bytes()
    if len(value) < 32:
        raise ValueError("Participant HMAC key must contain at least 32 bytes")
    return value


def _reference_markdown(case: dict) -> str:
    lines = ["### Ordered reference music"]
    for index, item in enumerate(case["reference_music"], 1):
        lines.append(f"{index}. {item['title']} — {item['artist']}")
    lines.append("\nListen to both candidates completely before rating them.")
    return "\n".join(lines)


def build_demo(
    study_dir: Path,
    database_path: Path,
    consent_text: str,
    participant_key: bytes,
):
    import gradio as gr

    cases = _load_cases(study_dir)
    case_ids = tuple(sorted(cases))
    connection = _connect(database_path)
    _initialize_database(connection)
    connection.close()

    def start_study(participant_code: str, consented: bool):
        if not consented:
            return None, None, "", "Please read and accept the consent statement.", {}
        try:
            digest = _participant_hash(participant_code, participant_key)
            with closing(_connect(database_path)) as database:
                assignment = _assign_participant(database, digest, case_ids)
            case = cases[assignment["case_id"]]
            display_generated_is_a = bool(assignment["display_song_a_is_generated"])
            base_generated_is_a = case["base_song_a_is_generated"]
            if display_generated_is_a == base_generated_is_a:
                path_a, path_b = case["path_a"], case["path_b"]
            else:
                path_a, path_b = case["path_b"], case["path_a"]
            state = {"session_id": assignment["session_id"]}
            return (
                str(path_a), str(path_b), _reference_markdown(case),
                "Candidates loaded. Complete every rating before submission.", state)
        except Exception as error:
            return None, None, "", f"Unable to start: {error}", {}

    def submit(
        state, fit_a, fit_b, quality_a, quality_b, novelty_a, novelty_b,
        preference, listening_freq, musical_training, playback_confirmed, notes,
    ):
        if not state or not state.get("session_id"):
            return "Please start the study before submitting."
        try:
            ratings = {
                "fit_a": fit_a, "fit_b": fit_b,
                "quality_a": quality_a, "quality_b": quality_b,
                "novelty_a": novelty_a, "novelty_b": novelty_b,
            }
            with closing(_connect(database_path)) as database:
                _submit_response(
                    database, state["session_id"], ratings, preference,
                    listening_freq, musical_training, playback_confirmed, notes)
            return "Thank you. Your anonymous response has been recorded."
        except Exception as error:
            return f"Unable to submit: {error}"

    with gr.Blocks(title="GenPlaylist Listener Study") as demo:
        gr.Markdown("# GenPlaylist Listener Study")
        gr.Markdown(consent_text)
        participant_code = gr.Textbox(
            label="Participant code", type="password",
            info="Use the code supplied by the recruitment platform.")
        consented = gr.Checkbox(label="I have read the statement and consent to participate")
        start = gr.Button("Start study")
        status = gr.Markdown()
        context = gr.Markdown()
        state = gr.State({})
        with gr.Row():
            audio_a = gr.Audio(label="Song A", interactive=False)
            audio_b = gr.Audio(label="Song B", interactive=False)
        gr.Markdown("Ratings: 1 = very poor, 5 = excellent")
        inputs = {}
        for dimension, label in (
            ("fit", "History fit"),
            ("quality", "Audio quality"),
            ("novelty", "Novelty"),
        ):
            with gr.Row():
                inputs[f"{dimension}_a"] = gr.Radio(
                    [1, 2, 3, 4, 5], label=f"Song A — {label}")
                inputs[f"{dimension}_b"] = gr.Radio(
                    [1, 2, 3, 4, 5], label=f"Song B — {label}")
        preference = gr.Radio(sorted(PREFERENCES), label="Overall preference")
        listening_freq = gr.Radio(
            sorted(LISTENING_FREQUENCIES), label="How often do you listen to music?")
        musical_training = gr.Radio(
            sorted(MUSICAL_TRAINING), label="Do you have formal musical training?")
        playback = gr.Checkbox(label="I played both candidate clips")
        notes = gr.Textbox(label="Optional comments", lines=3)
        submit_button = gr.Button("Submit ratings")
        submit_status = gr.Markdown()
        start.click(
            start_study,
            inputs=[participant_code, consented],
            outputs=[audio_a, audio_b, context, status, state],
        )
        submit_button.click(
            submit,
            inputs=[
                state,
                inputs["fit_a"], inputs["fit_b"],
                inputs["quality_a"], inputs["quality_b"],
                inputs["novelty_a"], inputs["novelty_b"],
                preference, listening_freq, musical_training, playback, notes,
            ],
            outputs=[submit_status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--consent-file", type=Path)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--participant-key", type=Path, default=None)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="verify the frozen package and exit without creating collection state",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.umask(0o077)
    study_dir = args.study_dir.expanduser().resolve()
    if args.validate_only:
        cases = _load_cases(study_dir)
        print(f"validated {len(cases)} blinded listener-study cases in {study_dir}")
        return 0
    if args.consent_file is None:
        raise ValueError("--consent-file is required unless --validate-only is used")
    consent_path = args.consent_file.expanduser().resolve()
    consent_text = consent_path.read_text(encoding="utf-8").strip()
    if not consent_text or "{{" in consent_text or "}}" in consent_text:
        raise ValueError("Consent text is empty or still contains template placeholders")
    database_path = (
        args.database.expanduser().resolve()
        if args.database else study_dir / "responses.sqlite3")
    key_path = (
        args.participant_key.expanduser().resolve()
        if args.participant_key else study_dir / ".participant_hmac_key")
    participant_key = _load_or_create_key(key_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    demo = build_demo(study_dir, database_path, consent_text, participant_key)
    if database_path.exists():
        database_path.chmod(0o600)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=False,
        show_error=False,
        allowed_paths=[str((study_dir / "participant_assets").resolve())],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

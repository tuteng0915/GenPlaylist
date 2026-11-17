# v2 retrieval-reference baselines and TIGER tuning — progress snapshot

Snapshot time: **2026-11-17 14:43 SGT**.  This record describes execution
state, not final metrics; it must not be cited as an outcome table.

## New no-cue retrieval-reference baseline

For every fixed 100-playlist cohort, the relevant retriever returns five
unique songs.  Each of the five audio samples for that playlist is conditioned
only on that same retrieved five-song reference set.  It does **not** read a
cue vocabulary, catalog cues, predicted cues, or a cue-derived prompt.

| Dataset | Retriever | Status at snapshot | Planned clips |
| --- | --- | --- | ---: |
| MPD | standard DDBC SID-only | Qwen reference verbalization in progress (36/500 records written) | 500 |
| Music4All | standard DDBC SID-only | queued after MPD arm | 500 |
| Music4All | corrected TIGER SID-only | queued after the DDBC arm | 500 |
| MPD | TIGER SID-only | withheld until the corrected MPD SID-only model finishes | 500 |

The first MPD DDBC clip passed the complete validation chain: it used the five
retrieved IDs as references, recorded an empty cue list, and passed waveform
decode, duration, seed, and content-hash checks.

## Corrected MPD TIGER SID-only control

The new MPD standard control is trained as **SID-only input -> SID-only
output**.  It is not substituted by the earlier invalid MPD TIGER output that
exposed catalog cues in its input.  At the snapshot it was at step 3,400 of
20,000 on a dedicated GPU; no recommendation number is reported yet.

## TIGER+cues warm-start optimization (Music4All)

Both finite follow-up trainings completed without test-set selection:

| Arm | Initialization | Steps | Cue-loss schedule | Checkpoint SHA-256 |
| --- | --- | ---: | --- | --- |
| `warm-ramp-0p02` | corrected SID-only checkpoint | 5,000 | 200-step cue-row bootstrap; ramp to 0.02 | `adef7dedf0cc28185aff10ca2df0924f9902805458be3e96e11f430cda8a60ee` |
| `warm-ramp-0p05` | corrected SID-only checkpoint | 5,000 | 200-step cue-row bootstrap; ramp to 0.05 | `0bba0c8984e7ca3b8d2ecb3b7d4f4e5c4bb14f80b48af799793a729cda12c0d4` |

Their fixed 941-context evaluation will report Recall@5, catalog diversity,
and maximum-weight one-to-one Cue F1 (overall, sound, emotion, theme).  The
evaluation was prepared but had not run at this snapshot because the permitted
two ASUS-01 GPUs were occupied by the MPD SID-only training and the no-cue
audio baseline.

## Resource boundary

Only two ASUS-01 GPUs are in use: GPU2 for the MPD TIGER SID-only control and
GPU3 for the no-cue retrieval-reference audio.  Existing ASUS-02 audio workers
are not modified by this follow-up.

# Reproduction audit — 2026-08-27

This audit records the recovery and reproduction completed after the server
data loss. All paths are relative to the repository root on `new-ncl`, whose
canonical location is `/home/wjzhang/tt_workspace/model/GenPlaylist`. Data,
checkpoints, generated audio, and result JSON files remain under the gitignored
`data/` tree unless stated otherwise.

## Frozen scope

- Protocol: one joint full-MASK 15-to-5 completion with a single sample per
  context; proxy evaluation uses full-catalog retrieval and multiset matching.
- MPD: 57,331 training windows and 941 frozen test histories.
- Music4All-Onion: 231,422 capped training windows and 4,725 user-disjoint test
  histories, with `K=5`, at least 8 unique references, at least 3 unique
  targets, cap 16, and repeated listens retained.
- Completed: six proxy methods on both datasets and the three-system MPD
  end-to-end audio experiment with MERT History Fit, VGGish FAD, and CLAP-A.
- Not included: the human listener study, Music4All end-to-end audio generation,
  and regeneration of the secondary RVQ-first end-to-end audio ablation.

## Dataset and cache identities

| Artifact | SHA-256 |
|---|---|
| MPD 0-cue prepared manifest | `d6f0158b2d1aed8ee73cd6a1af052b1a95df5470a6520121b3182c26142bcb7a` |
| MPD 8-cue prepared manifest | `a9a351da93e1dd95309e7c7b5e18da10730b3684ee42b747a8229faacc14bbd6` |
| Music4All train split | `81f5ee449c9a56831a957f432d10d0ba93acc6286bc79fd0b982b04c0e5eae6c` |
| Music4All test split | `cca5ce769ef87624a8dcf025cdc60efd9233b7043822d320ae976c53a1550652` |
| Music4All sequence manifest | `804a0f81006da53b950ac5936d0e5625e122efe17b9650f50cbd4200cfa3715d` |
| Music4All 0-cue prepared manifest | `9ed7b567f18144a07d9be1bb1c30514f4e53e4a4dc3a1b2f6827c9f75492dc9a` |
| Music4All 8-cue prepared manifest | `c751b55272ed9b631f56a94e4eb8eccf4a8adb7980277777767719fe794e16d1` |

The Music4All source scan contains 252,984,396 timestamped events from
119,140 users. Strict mapping retains 29,410,525 events over 2,754 catalog
items. The frozen split contains no validation partition: validation and test
were intentionally consolidated into the test set.

## Music4All checkpoint identities

| Model | SHA-256 |
|---|---|
| DDBC-SFT, 0 cues, step 20,000 | `8147394ad1a3c4648a32862138e66af79a023a2046686896660b87fcf3697f2a` |
| GenPlaylist, 8 cues, step 20,000 | `0232b8e96c4624df6bc524177863b63bbdfd0f77327cc4053ae45bba9f0f3c7a` |
| SASRec, step 20,000 | `cca47ae2591923a2c4bfd98dd91de95f8d58da8527908cb614a0bb158e565c6a` |
| TIGER, step 20,000 | `917cd6be5a099fe344a50cbb0b72d47074e6e66e7e0ca734aad5d163b2e1861e` |

## Legacy MPD comparison

The recovered MPD run was compared with the tracked legacy artifacts for
CLHE-kNN, SASRec, TIGER, DDBC-Base, DDBC-SFT, and GenPlaylist.

- All six `941 x 5` predicted item arrays are exactly equal.
- All six target arrays are exactly equal.
- Shared proxy metric fields have maximum absolute difference `0.0`.
- MERT metrics and confidence-interval endpoints have maximum absolute
  difference `5.960464477539063e-08`, attributable to float serialization.

The reproduced prediction JSON files live under
`data/results/reproduction-20260826/{mpd,music4all}`. Their complete hashes are
stored in the machine-readable audit at
`data/results/reproduction-20260826/reproduction-audit.json`.

## End-to-end audio audit

Qwen verbalization produced 941 records for each of ACE-Step-Direct, DDBC-SFT,
and GenPlaylist. All 2,823 records contain nonempty attributes and lyrics with
verse and chorus markup. ACE-Step then produced exactly 941 MP3 files and 941
metadata records for each system under matched seeds `42000 + history_index`.

| Artifact | SHA-256 |
|---|---|
| Verbalization manifest | `110af893bdb0a5660b3e017986e02361468145494253a6bfc72eb3f89eca8f91` |
| Verbalization audit | `94bd184764ae4ff3e0fc45934d951c17223cdc6c0a5e323e0e56bf9e4a2eec0a` |
| Audio manifest | `4807c9859debf2c4d7bd62463114f48814a62e6fa6f3ced68e438c29dd7d3ead` |
| MERT evaluation | `df552a583c7da745a7c34c30ca815205a8334c84d98730143d5bc7c2ebb6bd06` |
| VGGish FAD evaluation | `30bc0ec7f543261f6e5c668bb39e2b5919c6da6fc6d3d98bac3503963f9ba113` |
| Consolidated summary | `9efdd65d37c3b5314fa5c09552fc840aac8a03e1c90d2a963cd410ef0e9a4b47` |

| System | VGGish FAD down | MERT History Fit up | CLAP-A up |
|---|---:|---:|---:|
| ACE-Step-Direct | **4.9603** | **0.8448** | 0.2022 |
| DDBC-SFT | 5.2300 | 0.8426 | 0.2231 |
| GenPlaylist | 5.2457 | 0.8434 | **0.2284** |

The automatic metrics exactly reproduce the values already recorded in the
experiment ledger. MERT maximum-reference and maximum-catalog similarities are
retained as novelty/memorization diagnostics in `mert-evaluation.json`; no
additional standalone novelty score is introduced.

## Canonical machine-readable audit

The full generated audit is
`data/results/reproduction-20260826/reproduction-audit.json`, with SHA-256
`30b88babca8e3fff08086defb14ecb3d63b643b038061ace693dd5ff272870a4`.
It includes every reproduced prediction-result hash, checkpoint hash, audio
count and byte count, automatic metric artifact hash, and paired confidence
interval. The audit was generated from clean commit
`57133dfcac5ab29d29f1fad4262451634607f435`.

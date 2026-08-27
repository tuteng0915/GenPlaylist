# Frozen-protocol experiment ledger

This file records completed server runs under the authoritative
GenPlaylist-v4 protocol. It is an auditable working ledger, not a source of
claims for unfinished paper cells. All primary proxy results use 941 histories,
15 visible reference songs, five predicted songs, five targets, the complete
5,119-song catalog, and the frozen MERT revision in
`mert-v1-95m-catalog-v1`. Confidence intervals are 10,000-sample history-level
bootstrap intervals with seed 42.

The post-recovery artifact inventory, hashes, and exact legacy comparisons are
recorded in [`REPRODUCTION_AUDIT_20260827.md`](REPRODUCTION_AUDIT_20260827.md).

## Primary playlist-proxy results

| Method | N1-MERT | Recall@5 | M2M-MERT | Coverage@5 |
|---|---:|---:|---:|---:|
| CLHE-kNN | 0.8768 | 0.0674 | 0.8986 | 0.2510 |
| SASRec | **0.8795** | **0.1120** | **0.9004** | 0.4052 |
| DDBC-Base | 0.8723 | 0.0185 | 0.8915 | 0.4005 |
| DDBC-SFT (0 cues) | 0.8726 | 0.0223 | 0.8923 | **0.4055** |
| GenPlaylist (8 cues, curriculum) | 0.8722 | 0.0208 | 0.8916 | 0.3714 |
| TIGER adaptation (LR $10^{-3}$) | 0.8745 | 0.0550 | 0.8968 | 0.1909 |

The corresponding 95% intervals are:

| Method | N1-MERT | Recall@5 | M2M-MERT |
|---|---:|---:|---:|
| CLHE-kNN | [0.8739, 0.8796] | [0.0599, 0.0748] | [0.8971, 0.9000] |
| SASRec | [0.8759, 0.8829] | [0.1022, 0.1220] | [0.8989, 0.9020] |
| DDBC-Base | [0.8695, 0.8751] | [0.0145, 0.0227] | [0.8902, 0.8928] |
| DDBC-SFT (0 cues) | [0.8696, 0.8754] | [0.0181, 0.0268] | [0.8909, 0.8936] |
| GenPlaylist (8 cues, curriculum) | [0.8693, 0.8750] | [0.0170, 0.0249] | [0.8903, 0.8929] |
| TIGER adaptation (LR $10^{-3}$) | [0.8716, 0.8773] | [0.0478, 0.0625] | [0.8954, 0.8982] |

SASRec is currently the strongest proxy method. The intervals also show that
its Recall@5 improvement over CLHE-kNN is not explained by history sampling
noise. Conversely, the completed proxy runs do not support a claim that cue
tokens improve catalog continuation. Their value must be tested in the
end-to-end audio experiment rather than inferred from these rows.

## Music4All-Onion sequential results

The sequential experiment uses 231,422 capped training windows and one frozen
test window for each of 4,725 held-out users. Repeated listens are retained,
users are disjoint across train and test, and every example contains 15
references followed by five targets. DDBC-SFT and GenPlaylist use independently
trained 20,000-step Music4All checkpoints; the other four methods are rerun on
the same test set and catalog.

| Method | N1-MERT | Recall@5 | M2M-MERT | Coverage@5 |
|---|---:|---:|---:|---:|
| CLHE-kNN | 0.8831 | 0.1010 | **0.9000** | 0.4001 |
| SASRec | **0.8931** | **0.1261** | 0.8902 | 0.1911 |
| TIGER adaptation (LR $10^{-3}$) | 0.8852 | 0.0721 | 0.8824 | 0.1088 |
| DDBC-Base | 0.8709 | 0.0105 | 0.8871 | **0.6757** |
| DDBC-SFT (0 cues) | 0.8774 | 0.0841 | 0.8948 | 0.3391 |
| GenPlaylist (8 cues) | 0.8741 | 0.0624 | 0.8922 | 0.4401 |

These results do not show an 8-cue proxy advantage over the 0-cue SFT model:
DDBC-SFT is higher on Recall@5, N1-MERT, and M2M-MERT, while GenPlaylist has
higher catalog coverage. Cue utility is therefore evaluated at the audio
conditioning boundary rather than claimed from retrieval accuracy.

The TIGER adaptation generates one constrained semantic ID at a time and
recursively appends five predictions. Directly transferring the reported peak
learning rate of 0.01 collapses to only 22 unique tracks on this smaller
catalog (N1 0.8654, Recall@5 0.0066, M2M 0.8842). Lowering only the Adafactor
peak to 0.001 restores 977 unique predictions and produces the reported row.
This stronger adaptation is used in the main comparison; the collapsed run is
retained as a schedule diagnostic rather than evidence about TIGER itself.

## Cue and loss ablations

All rows below are separately trained 20,000-step checkpoints. `Full` is the
8-cue curriculum run. M2M is measured in MERT space; cue F1 is the auxiliary
frozen pseudo-label diagnostic.

| Variant | N1-MERT | Recall@5 | M2M-MERT | Coverage@5 | Cue F1 |
|---|---:|---:|---:|---:|---:|
| Full (8 cues) | 0.8722 | 0.0208 | 0.8916 | 0.3714 | 0.2832 |
| Without cues | 0.8726 | **0.0223** | **0.8923** | **0.4055** | N/A |
| 4 cues | 0.8731 | 0.0202 | 0.8915 | 0.3811 | 0.2794 |
| 16 cues | 0.8698 | 0.0155 | 0.8916 | 0.3598 | **0.2997** |
| 8 cues, uniform loss | 0.8713 | 0.0202 | 0.8911 | 0.3655 | 0.2782 |
| 8 cues, RVQ-first 0.05 to 0.25 | **0.8735** | 0.0196 | **0.8923** | 0.3919 | 0.2748 |

The default curriculum raises the per-cue weight from 0.1 to 1.0 by step
5,000. With eight cues, the final raw cue-weight sum is 8.0, whereas the three
RVQ layers plus collision token sum to 5.0. The RVQ-first run instead
uses a per-cue schedule of 0.05 to 0.25 through step 10,000, reducing the final
cue share from 61.5% to 28.6% before active-weight normalization.
RVQ-first improves N1-MERT, M2M-MERT, and coverage over the default curriculum,
but slightly lowers Recall@5 and cue F1. It therefore supports a loss-balance
effect, not a uniform improvement claim.

## Reproducibility anchors

- MERT model: `m-a-p/MERT-v1-95M`
- MERT revision: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`
- MERT manifest SHA-256:
  `4594f31205d646a8b9a69bb7f835840eb33cc6f56b677c8c1403a8d31d381bb7`
- CLHE-kNN prediction SHA-256:
  `3bac267feb33c8360d8b4f913cd4abf81c401d6ca15c49f2f0ea7d87063ade51`
- SASRec prediction SHA-256:
  `295fd6876f00bac7750a786bc4973fdc9cd3f6d11c302813a54e0317a47cab6a`
- TIGER LR-$10^{-3}$ prediction SHA-256:
  `8e4ca8858cb65901065b7270d9a39da3770d5a5d9750e823e885665cfe386b5b`
- DDBC-Base prediction SHA-256:
  `03a9447a7ea1058a55a375192982f0c8b9f11cded6b4a6484b671504886bc342`
- DDBC-SFT v4 plan SHA-256:
  `13fca255859809db6e7cca5bf0a6460ea1d4c1fa55ad10c26b23fd6e131922dc`
- GenPlaylist v4 plan SHA-256:
  `f341b77d8f61ec5849d36398967fb170570330ab5f01ed512e97ec555bf4710b`
- GenPlaylist RVQ-first prediction SHA-256:
  `275d8e8a52a684bb4724d5340270ad82698bea41919e6ac46903fe87881241b6`

The v4 WP-C result schema stores retrieved item IDs together with the exact
generated semantic token IDs and logical cue IDs. These plans are the only
valid inputs to the offline WP-D audio experiment; catalog cue labels must not
be substituted for GenPlaylist predictions.

## Frozen end-to-end audio protocol

The end-to-end run renders one 30-second waveform for each of the 941 histories
and each of three systems: ACE-Step-Direct, DDBC-SFT, and GenPlaylist. The
verbalizer is `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, using greedy decoding. The renderer
is `ACE-Step/ACE-Step-v1-3.5B` at revision
`82cd0d7b6322bd28cd4e830fe675ddb6180ce36c`, with 60 Euler steps. History
`i` uses diffusion seed `42000 + i` for all three systems. This matched-seed
design isolates the conditioning change from ACE-Step's initial noise.

The frozen output roots on the server are:

- Qwen3 conditions:
  `data/processed/genplaylist-end-to-end/qwen3-4b-verbalization-v1`
- ACE-Step waveforms:
  `data/processed/genplaylist-end-to-end/ace-step-v1`
- automatic metrics:
  `data/processed/genplaylist-end-to-end/metrics-v1`

The matched RVQ-first schedule ablation uses the parallel roots
`qwen3-rvq-first-v1`, `ace-step-rvq-first-v1`, and `metrics-rvq-first-v1`
under the same `genplaylist-end-to-end` directory. It changes only the trained
planner checkpoint; verbalizer, renderer, examples, durations, inference
steps, and per-history seeds remain fixed.

The main automatic audio metrics are VGGish FAD against the 941 immediate real
successors, MERT History Fit against the 15 visible references, and CLAP-A
cosine agreement with the generated music-attribute condition. MERT similarity
to the immediate successor, cross-history diversity, and maximum similarity to
the visible references or full catalog are retained as diagnostics. MERT uses
the same frozen revision recorded above. CLAP-A uses LAION-CLAP's non-fusion
`630k-audioset-best.pt` checkpoint at Hugging Face revision
`b3708341862f581175dba5c356a4ebf74a9b6651` (SHA-256
`8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`). Its
single 10-second crop uses the same `42000 + i` rule across systems. The WP-D
demo is not modified by these offline runners.

### Primary generated-audio results

| System | VGGish FAD ↓ | MERT History Fit ↑ | CLAP-A ↑ |
|---|---:|---:|---:|
| ACE-Step-Direct | **4.9603** | **0.8448** | 0.2022 |
| DDBC-SFT | 5.2300 | 0.8426 | 0.2231 |
| GenPlaylist | 5.2457 | 0.8434 | **0.2284** |

The history-level 95% intervals are:

| System | MERT History Fit | CLAP-A |
|---|---:|---:|
| ACE-Step-Direct | [0.8430, 0.8465] | [0.1960, 0.2083] |
| DDBC-SFT | [0.8409, 0.8443] | [0.2170, 0.2293] |
| GenPlaylist | [0.8416, 0.8451] | [0.2223, 0.2344] |

Paired bootstrap comparisons show no detectable History-Fit difference between
GenPlaylist and DDBC-SFT (difference 0.0007, 95% CI
[-0.0008, 0.0023]). GenPlaylist also does not significantly improve CLAP-A
over DDBC-SFT (difference 0.0053, 95% CI [-0.0027, 0.0130]). Both planned
systems improve CLAP-A over ACE-Step-Direct, whereas Direct obtains the best
FAD and raw History Fit. These results support a condition-adherence benefit
from preference planning in general, but do not yet isolate a benefit from
model-predicted cues over retrieved catalog cues.

The MERT diagnostics reinforce this interpretation. The Direct, DDBC-SFT, and
GenPlaylist next-song similarities are 0.8463, 0.8438, and 0.8445; their
cross-history diversities are 0.1025, 0.1002, and 0.1002. Maximum catalog
similarity is approximately 0.937 for all three, so none is uniquely flagged
by this memorization diagnostic.

All 2,823 verbalizations contain nonempty attributes and lyrics, and all lyric
drafts contain both verse and chorus markup. Fifteen attribute strings omit one
of the six requested comma-separated fields but remain valid ACE-Step prompts.
The default GenPlaylist plans contain at least one repeated cue phrase in 513
of 941 histories; this model behavior is retained and reported through cue
uniqueness rather than repaired after evaluation.

Reproducibility hashes:

- consolidated audio summary:
  `c2462816d415637edd93f2f7dec647041b166d7d46e882fd653a21d3261f3183`
- MERT audio evaluation:
  `d5ea744d08c089eb2e632a191abca09e49a859a411e1afc2e81fcb9a9bc9f184`
- VGGish FAD evaluation:
  `b292958b55fab11b0fa7bb7abbc5f6c5d99d2389ac94235ad47d4777c738c1c3`
- verbalization audit:
  `5084d1ddec711cc24f87ff48c51ca3f18701400d81b2251adf27e010240a5c90`

### RVQ-first generated-audio ablation

| Schedule | VGGish FAD ↓ | MERT History Fit ↑ | CLAP-A ↑ |
|---|---:|---:|---:|
| Default | 5.2457 | 0.8434 | **0.2284** |
| RVQ-first | **4.9170** | **0.8444** | 0.2241 |

Relative to Default, RVQ-first changes FAD by -0.3287. FAD is a corpus-level
point estimate, so no sample-level confidence interval is attached. The paired
History-Fit difference is 0.0010 (95% CI [-0.0004, 0.0025]), and the paired
CLAP-A difference is -0.0043 (95% CI [-0.0113, 0.0028]). Thus the ablation
improves the observed distributional-quality score but does not detectably
change history alignment or attribute adherence.

All 941 RVQ-first verbalizations pass the same nonempty and verse/chorus
checks. Three attribute strings contain fewer than six comma-separated fields.
The first generated plan repeats at least one cue phrase for 524 histories,
compared with 513 under Default; the schedule therefore does not resolve cue
duplication at the actual synthesis boundary.

RVQ-first reproducibility hashes:

- paired automatic comparison:
  `7a54a7fee88d133b9e4beed9bd247f928331414500e1232f6b914aef3a90d9ef`
- MERT manifest:
  `d78d7ae2381184627ca83cd22cb99a3a433de99fcb41b08dc1d97426343de5ae`
- CLAP-A evaluation:
  `04b7cf89dcc773728184bbe9079562405cb6f77b78d9872f10b945083f5b5954`
- VGGish FAD evaluation:
  `c74003ec2dbe27deb4f6d1a4e7be0f1c0700a3289061c999b31ace92aa2a1ebb`
- verbalization audit:
  `ef34da0465298afc0b6d1453addd1b73de3534a652369d92e3cc86e7ad8f0313`

## Frozen listener-study package

The offline study package at
`data/processed/genplaylist-listener-study-v1` uniformly samples 25 of the 941
frozen contexts without replacement using seed 42. Each case contains the 15
reference labels and a blinded A/B pair: the primary GenPlaylist waveform and
a centered clip of the immediate real next track. Both candidates are passed
through the same 30-second, 44.1-kHz, stereo PCM16 WAV conversion. GenPlaylist
occupies Song A in 13 cases and Song B in 12. All 50 WAV headers and hashes were
validated against the private manifest.

- participant-safe manifest SHA-256:
  `6d0c2d402069f34d47d889e38e42ca13ec67a1748b7c7366fe2bcc5e603ac995`
- private unblinding manifest SHA-256:
  `d9d1779b18a8f9884fe9236905e071f17c47ac59a01d7db7eca61848dcee7986`

These manifests freeze study materials only; no participant responses have
been collected. The private manifest, real-audio excerpts, and raw response
logs must not be published.

"""sweep_num_cues.py — reconstruction quality vs number of cues per song.

Assigns N cues per song (N over a grid, N may exceed the usual 6), decodes lyrics
from them, and scores the reconstruction against the real lyrics. Answers "does
giving the decoder more cues improve reconstruction, and where does it plateau?".

Additive / non-invasive: it does NOT use CueMappingEntry (which fixes 6 cues) — it
works with raw cue-string lists and feeds them to cue_eval.level3_reconstruction,
which already accepts an arbitrary {item_id: [cues]} mapping. The schema contract
(6 cues) is untouched.

Requires OPENAI_API_KEY (the decoder). One decode call per (N, song); cheap on
gpt-4o-mini and cached by prompt hash.

Usage:
  .venv/Scripts/python.exe sweep_num_cues.py --method tfidf --limit 300 \
      --eval-sample 60 --num-cues 0,3,6,9,12 \
      --min-df 5 --dedup-threshold 0.92 --lyrics-mode dedup --lyrics-cap 2000 --top-n 60
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src", "02_creative_cues"))
sys.path.insert(0, os.path.join(BASE, "src", "00_data_schema"))

import numpy as np       # noqa: E402
import cue_extractors    # noqa: E402
import cue_normalize     # noqa: E402
import cue_lyrics        # noqa: E402
import cue_eval          # noqa: E402
import cue_clients       # noqa: E402
import cue_io            # noqa: E402
from schema import CatalogItem  # noqa: E402

OUT_DIR = os.path.join(BASE, "src", "02_creative_cues", "outputs", "experiments")


def load_data(limit):
    with open(os.path.join(BASE, "data", "dataset", "catalog_metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    items = [CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
             for e in list(raw.values())[:limit]]
    lyrics = {}
    for it in items:
        p = os.path.join(BASE, "data", "lyrics", "spotify", f"{it.item_id}.txt")
        if os.path.isfile(p):
            lyrics[it.item_id] = open(p, encoding="utf-8", errors="ignore").read()
    return items, lyrics


def _song_text(item, lyrics, cap=600):
    meta = " ".join(p for p in [item.title, item.genre, item.mood, item.lyric_excerpt, *item.tags] if p)
    return f"{meta} {lyrics[:cap]}" if lyrics else meta


def select_n_cues(song_vec, vocab, valid_idx, valid_emb, n, lam=0.7, cand_k=40):
    """MMR selection of n cues for one song (mirrors cue_assign, any n).

    The candidate pool scales with n (>= max(cand_k, 2*n)) so N is never silently
    capped by the pool size — MMR always has more candidates than cues to pick.
    """
    if n <= 0 or len(valid_idx) == 0:
        return []
    rel = valid_emb @ song_vec                       # cosine to every valid cue
    k = min(len(valid_idx), max(cand_k, 2 * n))      # pool bigger than n (fixes false plateau)
    cand = np.argpartition(rel, -k)[-k:]
    cand = list(cand[np.argsort(rel[cand])[::-1]])
    selected: list[int] = []
    for _ in range(min(n, len(cand))):
        best_score, best = -1e9, None
        for c in cand:
            if c in selected:
                continue
            div = float((valid_emb[selected] @ valid_emb[c]).max()) if selected else 0.0
            score = lam * float(rel[c]) - (1 - lam) * div
            if score > best_score:
                best_score, best = score, c
        if best is None:
            break
        selected.append(best)
    return [vocab[int(valid_idx[c]) + 1] for c in selected]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--eval-sample", type=int, default=60)
    ap.add_argument("--num-cues", default="0,3,6,9,12", help="comma-separated cue counts")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--candidate-k", type=int, default=40,
                    help="base MMR candidate-pool size; effective pool = max(this, 2*N)")
    ap.add_argument("--score-chars", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not cue_clients.is_available():
        sys.exit("[sweep] OPENAI_API_KEY not set — the reconstruction decoder needs it.")

    ns = [int(x) for x in args.num_cues.split(",") if x.strip()]
    cue_eval.set_score_chars(args.score_chars)

    items, lyrics = load_data(args.limit)
    catalog_by_id = {it.item_id: it for it in items}
    lyrics_proc = {iid: cue_lyrics.preprocess_lyrics(t, args.lyrics_mode, args.lyrics_cap)
                   for iid, t in lyrics.items()}
    block = cue_normalize.build_block_tokens(items)
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)

    # vocab + cue embeddings (once)
    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=args.method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)
    norm = cue_normalize.build_vocab_normalized(
        raw, vocab_size=2048, min_df=args.min_df, dedup_threshold=args.dedup_threshold,
        block_tokens=block, verbose=True)
    vocab, cue_emb = norm["vocab"], norm["embeddings"]

    # valid (non-pad, non-zero) cue rows, normalized
    cue_norms = np.linalg.norm(cue_emb, axis=1)
    valid_idx = np.array([k for k in range(len(cue_emb))
                          if cue_norms[k] > 0 and not vocab[k + 1].startswith("<pad_")])
    if len(valid_idx) == 0:
        sys.exit("[sweep] no usable cues in the vocabulary.")
    valid_emb = cue_emb[valid_idx] / cue_norms[valid_idx][:, None]

    # song embeddings for the eval sample (once), same MiniLM space
    embed_fn, backend = cue_normalize._make_embedder()
    sample_ids = [iid for iid in catalog_by_id if lyrics.get(iid, "").strip()][:args.eval_sample]
    song_texts = [_song_text(catalog_by_id[iid], lyrics_proc.get(iid, "")) for iid in sample_ids]
    song_emb = embed_fn(song_texts)
    song_norms = np.linalg.norm(song_emb, axis=1, keepdims=True)
    song_emb = song_emb / np.where(song_norms == 0, 1.0, song_norms)
    print(f"[sweep] method={args.method} sample={len(sample_ids)} vocab(real)={len(valid_idx)} "
          f"grid N={ns} (embedder {backend})")

    results = {}
    for n in ns:
        cues_by_id = {}
        for r, iid in enumerate(sample_ids):
            cues_by_id[iid] = select_n_cues(song_emb[r], vocab, valid_idx, valid_emb, n,
                                            cand_k=args.candidate_k)
        print(f"[sweep] N={n}: decoding + scoring {len(sample_ids)} songs...")
        results[n] = cue_eval.level3_reconstruction(catalog_by_id, cues_by_id, lyrics, sample_ids)

    _write_report(results, ns, args, len(valid_idx), len(sample_ids))


def _write_report(results, ns, args, vocab_real, n_sample):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"sweep_numcues_{args.method}_{stamp}.md")
    lines = [
        f"# Reconstruction quality vs number of cues per song — `{args.method}`\n",
        f"_Generated {stamp} · {args.limit} songs · eval sample {n_sample} · vocab(real) {vocab_real} · "
        f"min_df {args.min_df} · lyrics-mode {args.lyrics_mode}_\n",
        "\nEach row assigns N cues per song, decodes lyrics from ONLY those cues (title/artist "
        "withheld), and scores vs the real lyrics. N=0 is the metadata-only floor. Higher is "
        "better for all columns except that BERTScore is baseline-rescaled (read rankings).\n",
        "\n| N cues | scored | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | Cosine |\n",
        "|--------|--------|---------|---------|---------|------|-----------|--------|\n",
    ]
    for n in ns:
        r = results.get(n, {})
        lines.append(f"| {n} | {r.get('n_scored','-')} | {r.get('rouge1_f','-')} | "
                     f"{r.get('rouge2_f','-')} | {r.get('rougeL_f','-')} | {r.get('bleu','-')} | "
                     f"{r.get('bertscore_f1','-')} | {r.get('sts_cosine','-')} |\n")
    lines += ["\n## How to read this\n",
              "- Expect quality to rise with N then plateau (or dip) once the decoder has enough "
              "signal; the plateau point suggests a good cue budget.\n",
              "- Cosine (document-level, length-robust) is the cleanest single column to track.\n",
              "- N=0 is the floor (no cues). The lift of N>0 over N=0 is the value of the cues.\n"]
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[sweep] report -> {path}")


if __name__ == "__main__":
    main()

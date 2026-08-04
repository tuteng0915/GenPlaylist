# WP-B — status and remaining TODO

## Implemented

- [x] Production default is the schema-compatible eight cues per song.
- [x] Keep 18 cues only as the explicit `research-18-cues` preset.
- [x] Atomically export vocabulary, mapping, and schema manifest.
- [x] Validate vocabulary, `<unk>`, cue ranges, IDs, and cue count.
- [x] Mark experimental artifacts as `wp_d_compatible=false`.

## Remaining experiments

- [ ] Run the eight-cue production build over all 5,119 catalog items.
- [ ] Check 100% ID coverage and archive artifact hashes.
- [ ] Audit cue quality and lyric leakage on a stratified sample.
- [ ] Freeze extraction and embedding model versions.

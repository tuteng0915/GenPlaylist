# WP-B — status and remaining TODO

## Implemented

- [x] Production stores 16 ranked cues per song; WP-C activates the first eight.
- [x] Keep 18 cues only as the explicit `research-18-cues` preset.
- [x] Atomically export vocabulary, mapping, and schema manifest.
- [x] Validate vocabulary, `<unk>`, cue ranges, IDs, and cue count.
- [x] Keep the legacy manifest field `wp_d_compatible` for artifact compatibility.
- [x] Build and hash the frozen 5,119-item production artifact.
- [x] Verify every frozen 20-song evaluation window has 16 stored cues per item.

## Remaining experiments

- [ ] Audit cue quality and lyric leakage on a stratified sample.
- [ ] Freeze extraction and embedding model versions.

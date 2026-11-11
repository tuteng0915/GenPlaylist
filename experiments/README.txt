GenPlaylist experiment snapshots (2026-11-11)
These are the executed experiment sources, including deployment-specific paths; not automatically launched on checkout.
- ddbc-preserve-20261111: alpha interpolation and SID-preserving joint training. Adopted alpha=0.2 uses the existing joint checkpoint, not the preservation arm.
- tiger-top5-20261111: same-checkpoint CPU AR5 vs one-pass beam top5 comparison; completed.
- list5-audio-20261111: 100 lists x5 songs x9 arms x2 datasets; 9000 clips complete. Known CLAP import failure for evaluate_mert_proxy remains in this exact executed snapshot. Fix and validate before restarting metrics; do not regenerate audio.
Final reports and provenance are in Odyssey-GenPlaylist/docs/results/experiment-status-20261111.

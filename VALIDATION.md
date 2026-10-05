# Release validation

- Seven unit tests pass: transition probabilities and conditional means; Bellman values against independent recursive enumeration; binary action and service conservation; deterministic seed replay; empirical record/window handling; delivery counts/public-state inference; policy eligibility/fallback behavior; batch filtering/order. (Several assertions are combined into individual test cases.)
- Twelve complete trajectories were independently re-simulated across full-history, 8-round and 24-round memory at held-out seeds 1003, 1049 and 1096. Every saved array agrees with release code within float32 archive tolerance (integer arrays exactly).
- Verified 480 evaluation archives, train/calibration/evaluation row counts, disjoint stage seeds and fresh feedback-evaluation seed ranges.
- All five final PDFs were rendered through Poppler and visually inspected. Legends, labels, intervals, footnotes and heatmaps were checked; PDF/SVG exports retain vector text and curves, with heatmaps rasterized deliberately.
- These checks validate implementation/reproducibility, not mathematical no-herding guarantees or model realism.

See `reproduce.py --mode verify` to repeat numerical checks. The release source checksum differs from the original locked checksum because optional diagnostic arguments were added after evaluation; replay validates the unchanged default numerical path.

## Proposal evidence audit (2026-10-04)

`audit_proposal.py` passed for all 96 nominal seeds and four policies (384 complete trajectories). Every saved array matched the fresh simulation, exactly for integer arrays and within float32 storage tolerance otherwise. Independent calculations reproduced load fractions, service completion, excess buildup, episode-wise worst-agent decision loss, and the paired bootstrap interval. The regenerated proposal PNG was pixel-identical to the existing proposal figure.

The representative episode is tied with other near-median episodes. Selection now explicitly chooses the smallest seed within 1e-12 of the minimum squared distance, preserving seed 1003 while avoiding floating-point tie instability. No simulator equations, trajectory values, or substantive results were changed. The proposal figure now exports inside this standalone repository by default; `--output-dir` permits a separate proposal destination.

This audit replays the entire nominal proposal comparison and selected mismatch trajectories, not the complete training, mismatch, and diagnostic study. `reproduce.py --mode full` rebuilds that broader study.

The documented figures pipeline completed successfully, including seven unit tests and 12 additional trajectory replays. The proposal plotting command also succeeded from a standalone copy outside the proposal directory. Dependency installation in a fresh environment was not repeated; commands were exercised with the existing pinned environment.

# Information timing and algorithmic herding

Reproducible numerical experiments in a synthetic, discrete-action stochastic resource-sharing game. These are **simulation results**, not field measurements or hand-drawn illustrations. The simulator computes agents' beliefs, Bellman action values, sampled actions, resource usage, and physical states. Plotting scripts read those saved arrays.

The proposal's Figure 1 is `figures/t1-mechanism-evidence.pdf`. Its two panels show action-value gaps and collective activation from the same matched episode. The original five supporting figures remain available.

## Install

Run from this repository's root. The recorded environment uses Python 3.13 and the exact versions in `requirements-lock.txt`.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Regenerate the exact proposal figure from saved data

```sh
.venv/bin/python src/proposal_figure_mechanism.py
```

This writes `figures/t1-mechanism-evidence.{pdf,svg,png}`. No parent proposal folder or LaTeX installation is required. The command reads:

- `configs/locked.json`: model and episode settings;
- `results/findings.json`: the representative-seed choice;
- `results/raw/evaluation/m00_s1003.npz`: the saved numerical trajectories.

To explicitly export into a separate proposal directory:

```sh
.venv/bin/python src/proposal_figure_mechanism.py --output-dir /path/to/proposal/figures
```

The heatmaps use `common_0__margin` and `staggered__margin`, transposed only for display. Margin is cost(wait) minus cost(activate). Gray cells indicate completed service, verified against backlog; the color range saturates at ±0.6. Panel B directly plots `load`, checked to equal the fraction of binary actions equal to one. There is no curve fitting, smoothing, or manual editing of trajectory values. Full-information follow-up starts at zero-based round 31; the vertical boundary is at 30.5.

Seed 1003 is selected near the median pair of common/staggered herding scores across 96 evaluation seeds. It is not selected for maximum improvement. Ties within absolute tolerance 1e-12 are resolved by the smallest seed. It is an illustrative episode; the reported statistics use all 96 seeds.

## Verify by running the simulator again

Quick numerical tests, split checks, and 12 saved-trajectory replays:

```sh
.venv/bin/python reproduce.py --mode verify
```

Full audit of the proposal's nominal evidence:

```sh
.venv/bin/python audit_proposal.py
```

The latter reruns all 96 held-out seeds (1001–1096), each with common periodic, group-staggered, common randomized, and frozen-belief policies: **384 trajectories**. It compares every output array with the saved archives, checks conservation and delivery counts, independently recomputes the cited metrics, reproduces the paired 4,000-resample bootstrap interval, and verifies the representative seed. It writes `results/proposal_audit.json` and exits with an error if a check fails. It does not overwrite original trajectories or findings. Allow roughly a few minutes, depending on hardware.

Integer arrays must match exactly. Saved floating arrays use float32; replay comparison allows `rtol=1e-6, atol=2e-6`. Recomputed metrics are compared to original float64 CSV values with 1e-12 tolerance. PDF metadata may differ between runs, so PDF byte equality alone is not a scientific reproducibility test.

## Regenerate all analyses and figures from saved data

```sh
.venv/bin/python reproduce.py --mode figures
```

This recomputes findings and reports, produces the five supporting figures and the proposal figure, and runs the quick verification checks. It overwrites derived outputs, but does not rerun the full simulation study.

## Rebuild the complete locked study

```sh
.venv/bin/python reproduce.py --mode full --workers 4
```

This reruns training, policy selection, calibration, evaluation, and independent diagnostic controls, then regenerates analysis, reports, figures, tests, and the checksum manifest. It is a substantially larger CPU job than the proposal-only audit. Preserve a copy of the release if comparing old and new results. The 384-trajectory audit covers nominal proposal evidence; it is not a replay of this entire study.

The exploratory pilot is separate:

```sh
.venv/bin/python src/pilot.py
```

The pilot does not automatically replace `configs/locked.json`. Use the locked config rather than `Config()` defaults, which differ from the selected experiment.

## Interpretation and limitations

Common randomized and staggered delivery both reduce peak activation relative to common periodic delivery in the locked model. Randomized delivery uses the same distribution for each agent, so this example does **not** establish that differentiated policies are necessary. The model was selected through an exploratory pilot; evaluation seeds are separate, but the results remain conditional on that model. The learner is an imperfect stochastic-FP-style empirical learner, not a demonstrated convergent stochastic FP algorithm. No potential-game property, analytical herding certificate, or field validity is established.

The supporting mismatch experiment does not reliably restore performance through the tested aggregate-feedback correction. That negative result is retained. Decision-loss summaries average each episode's worst-agent time-averaged reference loss; they are not a guarantee for every individual or episode.

## Files and scientific specification

- [MODEL.md](MODEL.md): game, observation access, learner, and metric definitions.
- [PROTOCOL.md](PROTOCOL.md): exploratory selection, independent splits, and inference.
- [RESULTS.md](RESULTS.md): numerical findings and limitations.
- [VALIDATION.md](VALIDATION.md): validation records.
- `configs/`: locked parameters, selected policies, calibrated detector.
- `src/model.py`: simulator and planning computations.
- `tests/`: Bellman recursion checked against exhaustive recursion, transition probabilities, conservation, record handling, and selection checks.
- `results/raw/evaluation/`: all 480 evaluation archives, including failures and all tested policies.
- `manifest.json`: SHA-256 file inventory and environment; regenerate with `python src/manifest.py` after final release preparation.

## GitHub release and proposal citation

Publish this folder as the repository root, excluding `.venv`, caches, and local temporary files. Keep the locked config, source, tests, analysis scripts, audit script, and documentation together. Include saved evaluation data in the repository or a versioned release archive so the fast plotting command works. The existing raw results occupy about 138 MB in total. Choose a license before release; this package does not assert an unapproved license.

Once the repository exists, make a versioned release and cite its verified URL, tag, and authors (or an archived release DOI). Do not cite a placeholder URL. The proposal's technical footnote can then point to that release for parameters, code, and reproduction instructions while the main text retains the model, learner, comparison, sample size, and limitations. No repository has been published by these scripts.

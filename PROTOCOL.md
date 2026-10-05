# Study protocol and provenance

## Exploration and locking

The pilot evaluates 32 parameter configurations (congestion 2/3, holding cost .01/.025, temperature .025/.05, prior mean .4/.6, delivery period 5/6) on seeds 101–112. Its complete results are retained. Configuration 17 was selected for a large common-minus-staggered excess-buildup difference, subject to period 6, staggered mean deadline completion >= .98 and mean worst-agent reference loss <= .04. This is **pilot-informed exploratory selection, not preregistration**. It intentionally finds a mechanism-demonstrating regime, not prevalence across arbitrary games.

Model parameters and empirical thresholds were locked before training, calibration and evaluation. Thresholds are H <= 1.25, reference loss <= .04, and deadline completion >= .98, all as sample means. Passing them is not certification or a per-agent probabilistic guarantee.

## Independent splits

| Stage | Seeds (inclusive) | Purpose |
|---|---|---|
| Pilot | 101–112 | Explore and select model regime |
| Training | 201–232 | Select policy and fit feature centers/scales |
| Calibration | 401–448 | Calibrate compatibility thresholds |
| Evaluation | 1001–1096 | Nominal policy comparisons, all five memory settings |
| Controls | 2001–2064 | Six independent robustness conditions |

Within a seed, competing policies share action uniforms, physical uniforms and agent heterogeneity. Randomized delivery has a separate random stream. Each policy generates its own endogenous actions, state and information records. Pairing reduces Monte Carlo variance without replaying one policy's outcomes as another's information.

Training/evaluation contain 25 policies plus a frozen-belief reference for each memory setting (full history, 4, 8, 16, 24 rounds). Calibration simulates only the training-selected nominal policy and frozen reference. All 480 evaluation seed-by-memory archives contain all 26 trajectories. Controls use three information policies and a matched frozen reference.

## Policy selection

The admissible design menu excludes full-information, public-state-only and frozen benchmarks because these change information frequency/access. It includes six common phases, 15 group-phase masks, fixed staggering and common randomized timing. Selection minimizes differentiation among policies meeting all three empirical training targets for every candidate memory. If no policy meets the H target, it chooses the lowest worst-model H among service/quality-eligible policies and records `training_target_met=false`. If no quality/service candidate exists, the broader fallback is explicitly flagged. No held-out outcomes choose the nominal, conservative or known-memory policy.

## Aggregate-feedback experiment

A feature vector uses six five-round averages of aggregate usage, six averages of public physical state, average absolute usage changes and peak usage. No private beliefs or true memory labels enter feedback selection. Centers/scales are fitted on eight training batches of four episodes; scale has floor .025. Each candidate's RMS standardized distance is calibrated on 12 independent four-episode batches. At alpha .1 the split-conformal rank is 12 of 12. Under exchangeability and a fixed candidate/probe rule, this gives marginal true-model retention at least 12/13; it is not simultaneous or adaptive-time coverage.

Evaluation seeds 1001–1048 form 12 probe batches; 1049–1096 form 12 fresh evaluation batches. Each inferred compatibility set selects a policy using training results only. Empty sets fall back to all candidate memories. Known-memory selection is a diagnostic benchmark with the true label and training data, not the hindsight optimum. This is one correction across reset episodes, not continuous online adaptation.

## Uncertainty and presentation

Means and paired differences use 4,000 percentile-bootstrap resamples with fixed seed 73. Feedback performance resamples 12 batch means, not 48 episodes as independent selection decisions. Detection rates use Wilson intervals over 12 decisions. Intervals are pointwise Monte Carlo uncertainty conditional on the locked model; they do not account for exploratory model selection or support population-wide generalization. Policy-frontier plots include every tested design, not just favorable ones.

The representative trajectory minimizes distance to the median pair of common/staggered H values over held-out seeds (seed 1003); it is illustrative, not independent confirmatory evidence. Numerical ties within absolute tolerance 1e-12 are resolved by the smallest seed. The same trajectory is used across its mechanism panels. Full numerical results remain available.

## Source provenance

`configs/locked.json:model_sha256` records the pre-evaluation model source. After evaluation, optional `ignore_public` and physical-override diagnostic support changed source bytes without changing the default model path. `verify.py` replays selected held-out trajectories under the release code. `manifest.json` records final source hashes. Both provenance stages are retained rather than replacing the original hash.

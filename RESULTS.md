# Results and proposal interpretation

All nominal means use 96 held-out seeds. Brackets are pointwise 95% bootstrap intervals. The model was selected in an earlier exploratory pilot; these are conditional simulator results, not universal evidence.

| Information rule | Excess buildup H | Peak active fraction | Worst-agent reference loss | Deadline completion | Total cost |
|---|---:|---:|---:|---:|---:|
| common_0 | 1.4987 [1.4388, 1.5556] | 0.9887 | 0.0165 | 0.9870 | 21.367 |
| staggered | 1.2235 [1.1836, 1.2652] | 0.8537 | 0.0076 | 0.9887 | 17.045 |
| randomized | 1.2027 [1.1628, 1.2422] | 0.8390 | 0.0083 | 0.9904 | 16.908 |
| full | 1.2786 [1.2326, 1.3247] | 0.8767 | 0.0028 | 0.9954 | 17.289 |
| public_only | 0.8668 [0.7713, 0.9627] | 0.9423 | 0.0461 | 0.9989 | 20.098 |
| frozen | 0.0000 [0.0000, 0.0000] | 0.9939 | 0.0626 | 0.9998 | 21.474 |

## Paired nominal contrasts

| Rule minus common periodic | Mean H difference | 95% paired interval |
|---|---:|---:|
| staggered | -0.2752 | [-0.3386, -0.2062] |
| randomized | -0.2960 | [-0.3633, -0.2252] |
| full | -0.2201 | [-0.2986, -0.1398] |
| public_only | -0.6319 | [-0.7418, -0.5208] |

## Feedback on fresh evaluation episodes

Each mean uses 12 independent probe/evaluation batches, four episodes on each side. Known-memory selection is trained, not an optimal oracle.

| Actual memory | Nominal H | Feedback H | Conservative H | Known-memory H |
|---|---:|---:|---:|---:|
| Full history | 1.2057 | 1.2405 | 1.2344 | 1.2057 |
| 24 | 1.2092 | 1.2569 | 1.2491 | 1.2092 |
| 16 | 1.2648 | 1.2821 | 1.3038 | 1.2595 |
| 8 | 1.3559 | 1.4288 | 1.4288 | 1.4288 |
| 4 | 1.2873 | 1.2743 | 1.2960 | 1.2873 |

## What this supports

- **Thrust 1:** a concrete finite-action example where information timing changes beliefs, action-value crossings and aggregate response. Staggering reduces nominal buildup and peak activation while retaining service. The mechanism panels expose the intermediate variables; they do not prove a general sufficient condition.
- **Thrust 2:** policy structure matters, and common randomized delivery is a serious zero-differentiation baseline. In this model, group-specific differentiation is not necessary to obtain the demonstrated reduction. Additional group-phase changes can hurt; finite-menu interactions do not establish submodularity.
- **Thrust 3:** memory mismatch changes performance and aggregate observations can leave several learner models plausible. The implemented one-step correction fails to deliver consistent recovery. This motivates the research problem but cannot be presented as successful preliminary correction evidence.

## Claims to avoid

No universal no-herding policy, mathematical certificate, alpha-potential structure, stochastic-FP convergence theorem, real feeder validation, welfare guarantee, or reliable mismatch recovery has been demonstrated. Low H alone can be misleading: frozen beliefs have H=0 by definition, and public-state-only information has low H but worse reference decision loss. Cost and service must accompany the herding diagnostic.

## Most useful next experimental changes

Vary initial backlogs and individual deadlines to test whether synchronization depends on shared service timing. Test a genuinely different signal family (for example a truthful forecast with calibrated uncertainty) rather than relabeling historical usage. Strengthen mismatch identification using designed probe policies, then test correction on a new held-out seed set; do not tune on the current test outcomes and retain confirmatory language.

"""Generate the numerical results document from saved analysis, without re-simulation."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];f=json.loads((R/'results/findings.json').read_text())
lines=['# Results and proposal interpretation','',
'All nominal means use 96 held-out seeds. Brackets are pointwise 95% bootstrap intervals. The model was selected in an earlier exploratory pilot; these are conditional simulator results, not universal evidence.','',
'| Information rule | Excess buildup H | Peak active fraction | Worst-agent reference loss | Deadline completion | Total cost |',
'|---|---:|---:|---:|---:|---:|']
for p in ['common_0','staggered','randomized','full','public_only','frozen']:
    v=f['summary']['0'][p];d=f['summary_ci']['0'][p]['herding']
    lines.append(f"| {p} | {v['herding']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] | {v['peak']:.4f} | {v['quality_worst']:.4f} | {v['deadline_completion']:.4f} | {v['mean_cost']:.3f} |")
lines+=['','## Paired nominal contrasts','', '| Rule minus common periodic | Mean H difference | 95% paired interval |','|---|---:|---:|']
for p,d in f['paired_H_difference_vs_common0'].items():
    lines.append(f"| {p} | {d['mean']:.4f} | [{d['lo']:.4f}, {d['hi']:.4f}] |")
lines+=['','## Feedback on fresh evaluation episodes','',
'Each mean uses 12 independent probe/evaluation batches, four episodes on each side. Known-memory selection is trained, not an optimal oracle.','',
'| Actual memory | Nominal H | Feedback H | Conservative H | Known-memory H |','|---|---:|---:|---:|---:|']
for m in [0,24,16,8,4]:
    vals=[f['feedback_ci'][str(m)][method]['herding']['mean'] for method in ['nominal','feedback','conservative','known_model']]
    lines.append('| '+('Full history' if m==0 else str(m))+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
lines+=['','## What this supports','',
'- **Thrust 1:** a concrete finite-action example where information timing changes beliefs, action-value crossings and aggregate response. Staggering reduces nominal buildup and peak activation while retaining service. The mechanism panels expose the intermediate variables; they do not prove a general sufficient condition.',
'- **Thrust 2:** policy structure matters, and common randomized delivery is a serious zero-differentiation baseline. In this model, group-specific differentiation is not necessary to obtain the demonstrated reduction. Additional group-phase changes can hurt; finite-menu interactions do not establish submodularity.',
'- **Thrust 3:** memory mismatch changes performance and aggregate observations can leave several learner models plausible. The implemented one-step correction fails to deliver consistent recovery. This motivates the research problem but cannot be presented as successful preliminary correction evidence.','',
'## Claims to avoid','',
'No universal no-herding policy, mathematical certificate, alpha-potential structure, stochastic-FP convergence theorem, real feeder validation, welfare guarantee, or reliable mismatch recovery has been demonstrated. Low H alone can be misleading: frozen beliefs have H=0 by definition, and public-state-only information has low H but worse reference decision loss. Cost and service must accompany the herding diagnostic.','',
'## Most useful next experimental changes','',
'Vary initial backlogs and individual deadlines to test whether synchronization depends on shared service timing. Test a genuinely different signal family (for example a truthful forecast with calibrated uncertainty) rather than relabeling historical usage. Strengthen mismatch identification using designed probe policies, then test correction on a new held-out seed set; do not tune on the current test outcomes and retain confirmatory language.','']
(R/'RESULTS.md').write_text('\n'.join(lines))
print('Wrote RESULTS.md')

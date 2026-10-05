"""Replay all 96 nominal seeds used by the proposal without changing study outputs."""
from pathlib import Path
import csv, hashlib, json, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from model import Config, Policy, simulate


def main():
    start = time.monotonic()
    lock = json.loads((ROOT / 'configs/locked.json').read_text())
    c = Config(**lock['model'])
    menu = [Policy('common_0'), Policy('staggered', (0, 1, 3, 4)),
            Policy('randomized', mode='randomized'), Policy('frozen', mode='frozen')]
    with (ROOT / 'results/evaluation.csv').open() as f:
        saved_rows = {(int(r['seed']), r['policy']): r for r in csv.DictReader(f) if int(r['memory']) == 0}
    findings = json.loads((ROOT / 'results/findings.json').read_text())
    results = {p.name: [] for p in menu}
    for count, seed in enumerate(range(*lock['seeds']['evaluation']), 1):
        fresh = {p.name: simulate(c, p, seed, memory=0) for p in menu}
        with np.load(ROOT / f'results/raw/evaluation/m00_s{seed:04d}.npz') as archive:
            for p in menu:
                tr = fresh[p.name]
                for key, value in tr.items():
                    stored = archive[p.name + '__' + key]
                    if value.dtype.kind == 'f':
                        np.testing.assert_allclose(value, stored, rtol=1e-6, atol=2e-6, equal_nan=True)
                    else:
                        np.testing.assert_array_equal(value, stored)
                # Independently calculate the proposal's metrics, without model.metrics.
                np.testing.assert_array_equal(tr['action'].sum(axis=0), c.demand - tr['backlog'][-1])
                np.testing.assert_allclose(tr['load'], tr['action'].sum(axis=1) / c.n)
                np.testing.assert_array_equal(np.isnan(tr['margin']), tr['backlog'][:-1] == 0)
                excess = np.r_[0., tr['load'] - fresh['frozen']['load']]
                score = max(0., max(excess[t] - excess[s] for t in range(1, len(excess)) for s in range(max(0, t-6), t)))
                row = dict(herding=score, peak=float(tr['load'].max()),
                           deadline_completion=float(tr['action'][:c.deadline].sum() / (c.n*c.demand)),
                           quality_worst=float(tr['quality_loss'].mean(axis=0).max()))
                for key, value in row.items():
                    np.testing.assert_allclose(value, float(saved_rows[seed, p.name][key]), rtol=1e-12, atol=1e-12)
                if p.mode != 'frozen':
                    np.testing.assert_array_equal(tr['delivery'][1:31].sum(axis=0), np.full(c.n, 5))
                    assert tr['delivery'][31:].all()
                results[p.name].append(row)
        if count % 16 == 0:
            print(f'Replayed {count}/96 seeds ({count*4} trajectories)', flush=True)
    summary = {name: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} for name, rows in results.items()}
    for name, row in summary.items():
        for key, value in row.items():
            np.testing.assert_allclose(value, findings['summary']['0'][name][key], rtol=1e-12, atol=1e-12)
    diff = np.array([s['herding']-c0['herding'] for s,c0 in zip(results['staggered'],results['common_0'])])
    rng = np.random.default_rng(73)
    boot = diff[rng.integers(96, size=(4000, 96))].mean(axis=1)
    interval = np.quantile(boot, [.025, .975])
    claimed = findings['paired_H_difference_vs_common0']['staggered']
    np.testing.assert_allclose([diff.mean(), *interval], [claimed['mean'],claimed['lo'],claimed['hi']],atol=1e-12)
    pairs = np.array([[a['herding'],b['herding']] for a,b in zip(results['common_0'],results['staggered'])])
    distance = ((pairs-np.median(pairs,axis=0))**2).sum(axis=1)
    # Equivalent arithmetic can reorder numerical ties; use the smallest tied seed.
    seed = lock['seeds']['evaluation'][0]+int(np.flatnonzero(np.isclose(distance, distance.min(), rtol=0, atol=1e-12))[0])
    assert seed == findings['representative_seed'] == 1003
    report = dict(status='PASS', trajectories=384, seeds=96, scope='Nominal proposal evidence; full mismatch/control study not replayed by this script',
                  summary=summary, staggered_minus_common=dict(mean=float(diff.mean()),ci95=interval.tolist()), representative_seed=seed,
                  model_sha256=hashlib.sha256((ROOT/'src/model.py').read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((ROOT/'configs/locked.json').read_bytes()).hexdigest(),
                  elapsed_seconds=round(time.monotonic()-start,2))
    (ROOT/'results/proposal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()

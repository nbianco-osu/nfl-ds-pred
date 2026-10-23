"""Matched calibration and paired week-bootstrap summaries, never promotion rules."""
import numpy as np


def evidence(report):
    rows=[r for r in report['games'] if 'evaluation' in r]
    calibration={}
    for model in ['markov','ridge_residual','poisson','ensemble']:
        pairs=[]
        for r in rows:
            if r['actual'][0]==r['actual'][1]:continue
            p=r['distributions'][model]
            denominator=p['home_win']+p['away_win']
            if denominator:pairs.append((p['home_win']/denominator,int(r['actual'][0]>r['actual'][1])))
        bins=[]
        for i in range(10):
            selected=[(p,y) for p,y in pairs if min(int(p*10),9)==i]
            if selected:bins.append(dict(lower=i/10,upper=(i+1)/10,n=len(selected),
                predicted=float(np.mean([p for p,y in selected])),observed=float(np.mean([y for p,y in selected]))))
        calibration[model]=bins
    comparisons=[]
    weeks=sorted({r['week'] for r in rows});rng=np.random.default_rng(42)
    for baseline in ['markov','ridge_residual','poisson']:
        for metric in ['brier3','log_loss3','margin_crps','total_crps']:
            sums=np.array([sum(r['evaluation']['ensemble'][metric]-r['evaluation'][baseline][metric] for r in rows if r['week']==week) for week in weeks])
            counts=np.array([sum(r['week']==week for r in rows) for week in weeks])
            if not weeks:continue
            draws=rng.integers(0,len(weeks),size=(2000,len(weeks)))
            delta=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
            comparisons.append(dict(baseline=baseline,metric=metric,games=len(rows),weeks=len(weeks),
                difference=float(sums.sum()/counts.sum()),interval95=np.quantile(delta,[.025,.975]).tolist()))
    return dict(calibration=calibration,paired=comparisons,
        note='Decisive-game calibration excludes ties. Paired season-week bootstrap, 2,000 resamples, seed 42. Few weeks and multiple comparisons limit inference; intervals are descriptive, not promotion evidence.')

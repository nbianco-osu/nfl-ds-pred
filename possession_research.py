"""Research scoring-event simulator; simplified possessions, not an NFL rules engine."""
import numpy as np
import pandas as pd


def crps(samples, actual):
    x=np.sort(np.asarray(samples,dtype=float));n=len(x)
    if not n or not np.isfinite(x).all():raise ValueError('Finite samples required')
    return float(np.abs(x-actual).mean()-np.sum((2*np.arange(1,n+1)-n-1)*x)/(n*n))


def parameters(pbp):
    drives=pbp.dropna(subset=['posteam','fixed_drive','fixed_drive_result']).drop_duplicates(['game_id','posteam','fixed_drive'])
    counts=drives.fixed_drive_result.value_counts()
    n=len(drives)
    if n<100 or drives.game_id.nunique()<16:raise ValueError('At least 16 earlier games and 100 drives required')
    outcomes=['Field goal','Touchdown','Opp touchdown','Safety']
    rates=np.array([(counts.get(k,0)+.5)/(n+3) for k in outcomes])
    conversions=np.array([int(pbp.extra_point_result.isin(['failed','blocked']).sum()+pbp.two_point_conv_result.eq('failure').sum())+1,
                          int(pbp.extra_point_result.eq('good').sum())+1,
                          int(pbp.two_point_conv_result.eq('success').sum())+1],dtype=float)
    conversions/=conversions.sum()
    return dict(rates=rates,td_points=conversions,drives=float(drives.groupby(['game_id','posteam']).size().mean()),games=int(drives.game_id.nunique()))


def simulate(expected, params, trials=2000, seed=42):
    expected=np.asarray(expected,dtype=float)
    if expected.shape!=(2,) or not np.isfinite(expected).all() or (expected<0).any():raise ValueError('Two finite nonnegative expected scores required')
    rng=np.random.default_rng(seed);pace=max(params['drives'],1)
    count=np.maximum(1,rng.poisson(pace,size=trials))
    scores=np.zeros((trials,2),dtype=int)
    td_mean=np.dot([6,7,8],params['td_points'])
    side_rates=[]
    for target in expected:
        rates=np.array(params['rates'],copy=True)
        own_mean=3*rates[0]+td_mean*rates[1]
        scale=max(0,(target/pace-td_mean*rates[2]-2*rates[3]))/max(own_mean,1e-8)
        rates[:2]*=scale
        if rates.sum()>.95:rates[:2]*=max(0,.95-rates[2:].sum())/max(rates[:2].sum(),1e-8)
        side_rates.append(np.r_[1-rates.sum(),rates])
    def possession(indices,side):
        outcomes=rng.choice(5,size=len(indices),p=side_rates[side])
        scores[indices,side]+=3*(outcomes==1)
        td=rng.choice([6,7,8],size=len(indices),p=params['td_points'])
        scores[indices,side]+=td*(outcomes==2)
        scores[indices,1-side]+=td*(outcomes==3)+2*(outcomes==4)
    for drive in range(int(count.max())):
        indices=np.flatnonzero(count>drive)
        for side in [0,1]:possession(indices,side)
    # Bounded OT approximation: both sides receive a possession in each pair.
    # At most two pairs; remaining ties are retained, not forced to a winner.
    for _ in range(2):
        tied=np.flatnonzero(scores[:,0]==scores[:,1])
        if not len(tied):break
        for side in rng.permutation([0,1]):possession(tied,int(side))
    return scores


def evaluate(pbp, frame, model, cols):
    results=[];folds=[]
    for week,rows in frame.groupby('week'):
        if week<3:continue
        earlier=pbp.loc[pbp.week.lt(week)]
        try:params=parameters(earlier)
        except ValueError:continue
        folds.append(dict(test_week=int(week),calibration_last_week=int(earlier.week.max()),
                          games=params['games'],drives=params['drives'],rates=params['rates'].tolist(),td_points=params['td_points'].tolist()))
        expected=model.predict(rows[cols])
        for i,game in enumerate(rows.to_dict('records')):
            seed=42+int(week)*1000+i
            event=simulate(expected[i],params,seed=seed)
            independent=np.random.default_rng(seed).poisson(expected[i],size=(2000,2))
            actual=np.array([game['home_score'],game['away_score']])
            item=dict(game_id=game['game_id'],week=int(week),calibration_games=params['games'])
            for name,draws in [('possession',event),('poisson',independent)]:
                item[name+'_total_crps']=crps(draws.sum(axis=1),actual.sum())
                item[name+'_margin_crps']=crps(draws[:,0]-draws[:,1],actual[0]-actual[1])
            results.append(item)
    return dict(status='research_only',games=len(results),rows=results,folds=folds,
        mean_metrics={key:float(np.mean([r[key] for r in results])) for key in ['possession_total_crps','poisson_total_crps','possession_margin_crps','poisson_margin_crps']} if results else {},
        note='2026 revised-source research. Each week learns drive outcomes/extra-point mixtures from earlier weeks only; at least 16 earlier games. '
             'Shared Poisson pace, integer FG/TD/conversion/defensive-score events; two paired OT opportunities are an approximation, not complete clock/field-position/rules simulation. '
             'Compared to independent Poisson on identical expected scores and games. Seed42; 2000 trials. No production activation or claim of better future accuracy.')

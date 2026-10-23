"""Regulation-only empirical semi-Markov research. Never writes production state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from possession_research import crps


def extract_drives(pbp):
    """Keep observed score increments and duration jointly, including defensive scores."""
    required = ['game_id', 'season', 'week', 'play_id', 'qtr', 'fixed_drive',
                'posteam', 'defteam', 'game_seconds_remaining', 'yardline_100',
                'posteam_score', 'defteam_score', 'posteam_score_post', 'defteam_score_post']
    missing = set(required) - set(pbp.columns)
    if missing:
        raise ValueError(f'Missing PBP columns: {sorted(missing)}')
    p = pbp.loc[pbp.qtr.between(1, 4)].sort_values(['game_id', 'play_id'])
    rows = []
    for (game, drive), group in p.dropna(subset=['posteam', 'fixed_drive']).groupby(['game_id', 'fixed_drive'], sort=False):
        # Kickoff yardlines describe the kicking team, not the receiving offense's start.
        if 'play_type' in group:
            group = group.loc[~group.play_type.isin(['kickoff', 'no_play'])]
        group = group.dropna(subset=required[6:])
        if group.empty or group.posteam.nunique() != 1:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        own = last.posteam_score_post - first.posteam_score
        opp = last.defteam_score_post - first.defteam_score
        duration = first.game_seconds_remaining - last.game_seconds_remaining
        if own not in [0, 2, 3, 6, 7, 8] or opp not in [0, 2, 6, 7, 8] or duration < 0:
            continue
        rows.append(dict(game_id=game, drive=int(drive), season=int(first.season), week=int(first.week),
                         offense=first.posteam, defense=first.defteam, remaining=float(first.game_seconds_remaining),
                         margin=float(first.posteam_score-first.defteam_score), field=float(first.yardline_100),
                         duration=max(1, int(duration)), own=int(own), opp=int(opp), next_field=70., retain=False))
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No usable regulation drives')
    for _, indices in result.groupby('game_id', sort=False).groups.items():
        indices = list(indices)
        for i, j in zip(indices, indices[1:]):
            a, b = result.loc[i], result.loc[j]
            if (a.remaining > 1800) == (b.remaining > 1800):
                result.loc[i, 'next_field'] = b.field
                result.loc[i, 'retain'] = a.offense == b.offense
    return result


def fit(drives, season, week, team_weight=0.0):
    prior = drives.loc[(drives.season < season) | ((drives.season == season) & (drives.week < week))].copy()
    if prior.game_id.nunique() < 16 or len(prior) < 100:
        raise ValueError('Need 16 earlier games and 100 drives')
    if not 0 <= team_weight <= 1:
        raise ValueError('team_weight must be in [0,1]')
    return {'drives': prior, 'team_weight': team_weight, 'cutoff': [int(season), int(week)]}


def simulate(model, home, away, trials=1000, seed=42):
    """Sample a joint duration/outcome/next-field transition, bounded by half clocks."""
    if not isinstance(trials, int) or trials < 1 or home == away:
        raise ValueError('Positive integer trials and distinct teams required')
    rng = np.random.default_rng(seed)
    d = model['drives']
    fields, margins, remaining = (d[k].to_numpy() for k in ['field', 'margin', 'remaining'])
    records = d[['duration', 'own', 'opp', 'next_field', 'retain']].to_numpy(dtype=float)
    scores = np.zeros((trials, 5, 2), dtype=int)
    # Discrete state pools avoid constructing a full transition tensor from sparse data.
    cache = {}
    for trial in range(trials):
        score = np.zeros(2, dtype=int)
        first_receiver = int(rng.integers(2))
        for half in range(2):
            side, field, clock = (first_receiver if half == 0 else 1-first_receiver), 70., 1800
            while clock > 0:
                global_clock = clock + (1800 if half == 0 else 0)
                margin = int(score[side]-score[1-side])
                key = (side, int(field//25), int(np.clip(margin//7, -3, 3)), global_clock <= 300, half)
                if key not in cache:
                    distance = abs(fields-(key[1]*25+12.5))/25 + abs(np.clip(margins, -21, 21)-key[2]*7)/7
                    distance += 2*((remaining <= 300) != (global_clock <= 300))
                    weights = np.exp(-distance) + .05  # League smoothing keeps rare states supported.
                    offense, defense = (home, away) if side == 0 else (away, home)
                    team = (d.offense.eq(offense) | d.defense.eq(defense)).to_numpy()
                    local = weights*team
                    weights /= weights.sum()
                    if local.sum():
                        weights = (1-model['team_weight'])*weights + model['team_weight']*local/local.sum()
                    cache[key] = np.cumsum(weights)
                    cache[key][-1] = 1.
                duration, own, opp, next_field, retain = records[np.searchsorted(cache[key], rng.random())]
                end = max(0, clock-int(duration))
                # A sampled drive that overruns the half is censored without awarding its score.
                if duration <= clock:
                    score[side] += int(own); score[1-side] += int(opp)
                elapsed = half*1800 + 1800-end
                old_elapsed = half*1800 + 1800-clock
                for quarter in range(1, 5):
                    boundary = quarter*900
                    if old_elapsed < boundary <= elapsed:
                        before = score.copy()
                        if duration <= clock and boundary < elapsed:
                            before[side] -= int(own); before[1-side] -= int(opp)
                        scores[trial, quarter] = before
                clock = end; field = next_field
                if not retain:
                    side = 1-side
    return scores


def probabilities(paths):
    margin = paths[:, -1, 0]-paths[:, -1, 1]
    return np.array([(margin > 0).mean(), (margin == 0).mean(), (margin < 0).mean()])


def simulate_batch(model, home, away, trials=1000, seed=42):
    """Vectorized trial scheduling with the same coarse state/transition assumptions."""
    if not isinstance(trials,int) or trials<1 or home==away:
        raise ValueError('Positive integer trials and distinct teams required')
    rng=np.random.default_rng(seed);d=model['drives']
    fields=d.field.to_numpy();margins=np.clip(d.margin.to_numpy(),-21,21)
    late=d.remaining.to_numpy()<=300
    transitions=d[['duration','own','opp','next_field','retain']].to_numpy(dtype=int)
    local=[(d.offense.eq(o)|d.defense.eq(a)).to_numpy() for o,a in [(home,away),(away,home)]]
    cdfs={};scores=np.zeros((trials,2),int);paths=np.zeros((trials,5,2),int)
    first=rng.integers(2,size=trials)
    for half in range(2):
        side=first.copy() if half==0 else 1-first
        field=np.full(trials,70);clock=np.full(trials,1800)
        while np.any(clock>0):
            ix=np.flatnonzero(clock>0)
            margin=scores[ix,side[ix]]-scores[ix,1-side[ix]]
            keys=np.column_stack([side[ix],field[ix]//25,np.clip(margin//7,-3,3),(clock[ix]+(1800 if half==0 else 0))<=300])
            unique,inverse=np.unique(keys,axis=0,return_inverse=True)
            selected=np.empty(len(ix),int)
            for j,keyarray in enumerate(unique):
                key=tuple(keyarray)
                if key not in cdfs:
                    dist=abs(fields-(key[1]*25+12.5))/25+abs(margins-key[2]*7)/7+2*(late!=key[3])
                    w=np.exp(-dist)+.05;loc=w*local[key[0]];w/=w.sum()
                    if loc.sum():w=(1-model['team_weight'])*w+model['team_weight']*loc/loc.sum()
                    cdfs[key]=np.cumsum(w);cdfs[key][-1]=1.
                mask=inverse==j
                selected[mask]=np.searchsorted(cdfs[key],rng.random(mask.sum()))
            duration,own,opp,next_field,retain=transitions[selected].T
            old_clock=clock[ix].copy();end=np.maximum(0,old_clock-duration)
            before=scores[ix].copy();valid=duration<=old_clock
            scores[ix,side[ix]]+=own*valid;scores[ix,1-side[ix]]+=opp*valid
            elapsed=half*1800+1800-end;old_elapsed=half*1800+1800-old_clock
            for quarter in range(1,5):
                boundary=quarter*900;cross=(old_elapsed<boundary)&(elapsed>=boundary)
                paths[ix[cross],quarter]=np.where((elapsed[cross]==boundary)[:,None],scores[ix[cross]],before[cross])
            clock[ix]=end;field[ix]=next_field;side[ix]=np.where(retain,side[ix],1-side[ix])
    return paths


def choose_weight(previous):
    """Select only from earlier out-of-fold regulation outcomes; no evaluation labels."""
    if len(previous) < 16:
        return .5, 'Equal-weight fallback: fewer than 16 earlier out-of-fold games'
    losses = []
    for weight in np.linspace(0, 1, 11):
        loss = np.mean([np.sum((weight*np.array(r['team'])+(1-weight)*np.array(r['league'])-np.eye(3)[r['outcome']])**2) for r in previous])
        losses.append((loss, abs(weight-.5), float(weight)))
    return min(losses)[2], 'Weight selected by earlier-week three-outcome Brier loss'


def mix_paths(league, team, weight, seed=42):
    if league.shape != team.shape or not 0 <= weight <= 1:
        raise ValueError('Matched path arrays and valid mixture weight required')
    use_team = np.random.default_rng(seed).random(len(league)) < weight
    return np.where(use_team[:, None, None], team, league)


def summarize(paths):
    return {'home_regulation_win':float(probabilities(paths)[0]), 'regulation_tie':float(probabilities(paths)[1]),
            'away_regulation_win':float(probabilities(paths)[2]),
            'quarters':[{'quarter':q, 'home_mean':float(paths[:,q,0].mean()), 'away_mean':float(paths[:,q,1].mean()),
                         'home_interval80':np.quantile(paths[:,q,0],[.1,.9]).tolist(),
                         'away_interval80':np.quantile(paths[:,q,1],[.1,.9]).tolist(),
                         'home_leading':float((paths[:,q,0]>paths[:,q,1]).mean())} for q in range(1,5)],
            'sample_paths':paths[:12].tolist()}


def run(pbp, schedule, season, trials=1000):
    drives = extract_drives(pbp)
    completed = schedule.loc[schedule.season.eq(season) & schedule.game_type.eq('REG')].sort_values(['week','game_id'])
    final = pbp.loc[pbp.qtr.eq(4)].sort_values(['game_id','play_id']).groupby('game_id').tail(1).set_index('game_id')
    previous, games, skipped = [], [], []
    for week, batch in completed.groupby('week'):
        try:
            league = fit(drives, season, week)
            team = fit(drives, season, week, .35)
        except ValueError as error:
            skipped.extend({'game_id':g, 'reason':str(error)} for g in batch.game_id); continue
        weight, reason = choose_weight(previous)
        week_results = []
        for game in batch.itertuples():
            seed = int(hashlib.sha256(game.game_id.encode()).hexdigest()[:8],16)
            a = simulate(league, game.home_team, game.away_team, trials, seed)
            b = simulate(team, game.home_team, game.away_team, trials, seed+1)
            paths = mix_paths(a,b,weight,seed+2)
            record = dict(game_id=game.game_id, week=int(week), home_team=game.home_team, away_team=game.away_team,
                          team_weight=weight, weight_basis=reason, weight_training_games=len(previous),
                          training_games=int(league['drives'].game_id.nunique()), training_max_week=int(league['drives'].week.max()),
                          **summarize(paths))
            if game.game_id in final.index and pd.notna(game.home_score):
                actual = final.loc[game.game_id, ['total_home_score','total_away_score']].to_numpy(dtype=float)
                if np.isfinite(actual).all():
                    outcome = 0 if actual[0]>actual[1] else 1 if actual[0]==actual[1] else 2
                    week_results.append(dict(league=probabilities(a).tolist(),team=probabilities(b).tolist(),outcome=outcome))
                    record['evaluation'] = {'actual_regulation_score':actual.tolist(),
                        'brier3':float(np.sum((probabilities(paths)-np.eye(3)[outcome])**2)),
                        'total_crps':crps(paths[:,-1].sum(axis=1),actual.sum()),
                        'margin_crps':crps(paths[:,-1,0]-paths[:,-1,1],actual[0]-actual[1])}
            games.append(record)
        previous.extend(week_results)
    return dict(status='retrospective_research_only', scope='regulation_only', games=games, skipped=skipped,
                trials=trials, season=season, source_games=int(drives.game_id.nunique()), source_max_week=int(drives.week.max()),
                note='Revised-source replay, not saved pregame forecasts. Regulation ties are retained; no overtime, final-game winner, spread or total grading. Quarter bands are score/lead trajectories, not calibrated in-game win probabilities. No production promotion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pbp', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing research run')
    pbp = pd.read_parquet(args.pbp)
    report = run(pbp, pd.read_csv(args.schedule), args.season, args.trials)
    report['sources'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.pbp,args.schedule]}
    args.output.mkdir(parents=True)
    extract_drives(pbp).to_csv(args.output/'transitions.csv',index=False)
    (args.output/'model.json').write_text(json.dumps({'schema_version':1,'model':'empirical_semi_markov',
        'scope':'regulation_only','team_conditioning_weight':.35,'league_smoothing':.05,
        'sources':report['sources'],'fit_protocol':'Only season/week strictly before target; mixture weights from earlier out-of-fold games'},indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'games':len(report['games']),'skipped':len(report['skipped']),'output':str(args.output)}))

"""Shared latent team-strength season draws with preserved matchup marginals."""
import numpy as np
import pandas as pd
from scipy.special import ndtr


def simulate(schedule, predictions, trials=10000, seed=42, strength_sd=.35):
    if trials < 1 or strength_sd < 0 or schedule.game_id.duplicated().any():
        raise ValueError('Invalid season simulation inputs')
    teams = sorted(set(schedule.home_team) | set(schedule.away_team))
    index = {t:i for i,t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, strength_sd, (trials, len(teams)))
    wins = np.zeros_like(strength); ties = np.zeros_like(strength)
    paths = {t:[] for t in teams}
    expected_decisive = np.zeros(trials); expected_ties = np.zeros(trials)
    for week, games in schedule.sort_values(['week','game_id']).groupby('week'):
        for g in games.itertuples():
            h,a = index[g.home_team],index[g.away_team]
            if pd.notna(g.home_score) and pd.notna(g.away_score):
                home = np.full(trials, g.home_score > g.away_score)
                tie = np.full(trials, g.home_score == g.away_score)
            else:
                if g.game_id not in predictions:
                    raise ValueError(f'Missing future distribution: {g.game_id}')
                p = predictions[g.game_id]
                probs = np.array([p['home_win'],p['tie'],p['away_win']])
                if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(),1):
                    raise ValueError('Invalid three-outcome probabilities')
                # A Gaussian copula correlates team seasons without shifting input marginals.
                u = ndtr((strength[:,h]-strength[:,a]+rng.normal(size=trials))/np.sqrt(1+2*strength_sd**2))
                home = u < probs[0]; tie = (u >= probs[0]) & (u < probs[0]+probs[1])
            away = ~(home | tie)
            wins[:,h] += home; wins[:,a] += away
            ties[:,h] += tie; ties[:,a] += tie
            expected_decisive += ~tie; expected_ties += tie
        for t,i in index.items():
            paths[t].append(dict(week=int(week), mean_wins=float(wins[:,i].mean()),
                low=float(np.quantile(wins[:,i],.1)),high=float(np.quantile(wins[:,i],.9)),
                mean_ties=float(ties[:,i].mean())))
    if not np.array_equal(wins.sum(axis=1),expected_decisive) or not np.array_equal(ties.sum(axis=1),2*expected_ties):
        raise ValueError('Season conservation failure')
    return dict(trials=trials,seed=seed,strength_sd=strength_sd,teams=paths,conservation_passed=True,
                note='Shared Gaussian team-strength copula; SD 0.35 is an assumption, not fitted. Preserves input game marginals. Completed results fixed; wins exclude ties. No playoffs or arbitrary win caps.')

"""Reproducible noisy season simulations, with completed results fixed."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = {'simulations': 10000, 'seed': 42, 'logit_noise_sd': 0.35, 'interval': 0.8}


def simulate(rows, simulations=10000, seed=42, logit_noise_sd=0.35):
    if rows.game_id.duplicated().any() or not rows.home_win_probability.between(0, 1).all():
        raise ValueError('Require unique games and valid probabilities')
    if simulations < 1 or logit_noise_sd < 0:
        raise ValueError('Invalid simulation settings')
    rows = rows.sort_values('game_id')
    rng = np.random.default_rng(seed)
    teams = sorted(set(rows.home_team) | set(rows.away_team))
    wins = {team: np.zeros(simulations, dtype=int) for team in teams}
    played = {team: 0 for team in teams}
    for game in rows.itertuples():
        if game.status == 'Final':
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                raise ValueError('Final game lacks scores')
            wins[game.home_team] += int(game.home_score > game.away_score)
            wins[game.away_team] += int(game.away_score > game.home_score)
            played[game.home_team] += int(game.home_score > game.away_score)
            played[game.away_team] += int(game.away_score > game.home_score)
        else:
            p = np.clip(game.home_win_probability, 1e-6, 1 - 1e-6)
            noisy_logit = np.log(p / (1 - p)) + rng.normal(0, logit_noise_sd, simulations)
            probability = 1 / (1 + np.exp(-noisy_logit))
            home_win = rng.random(simulations) < probability
            wins[game.home_team] += home_win
            wins[game.away_team] += ~home_win
    return pd.DataFrame([{'team': team, 'actual_wins': played[team],
                          'expected_wins': float(values.mean()),
                          'low_wins': int(np.quantile(values, .1, method='nearest')),
                          'high_wins': int(np.quantile(values, .9, method='nearest')),
                          'probability_16_plus': float((values >= 16).mean())}
                         for team, values in wins.items()]).sort_values('expected_wins', ascending=False)


def export_simulations(rows, root):
    summary = simulate(rows, **{k: v for k, v in CONFIG.items() if k != 'interval'})
    metadata = pd.read_csv(root / 'data/team_metadata.csv').set_index('team_abbr')
    summary['team_name'] = summary.team.map(metadata.team_name)
    result = {'config': CONFIG, 'teams': json.loads(summary.to_json(orient='records')),
              'note': 'Assumed noise, not calibrated uncertainty. Completed results fixed; remaining games sampled independently. Ranges are simulation percentiles, not guarantees.'}
    (root / 'public_site/data/season_simulations.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (root / 'models/simulation_config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    print(export_simulations(pd.read_csv(root / 'predictions/nfl_2026_predictions_advanced_no_market_deep.csv'), root).to_string(index=False))

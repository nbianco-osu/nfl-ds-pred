"""Weekly matchup probabilities and cumulative team win trajectories."""
import json
from pathlib import Path


def build_series(predictions):
    ids = [row['game_id'] for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate games')
    teams = sorted({row[side + '_team'] for row in predictions for side in ['home', 'away']})
    last_complete = max((row['week'] for row in predictions if row.get('status') == 'Final'), default=0)
    result = []
    for team in teams:
        games = [row for row in predictions if team in [row['home_team'], row['away_team']]]
        first = games[0]
        side = 'home' if team == first['home_team'] else 'away'
        by_week = {row['week']: row for row in games}
        if len(by_week) != len(games):
            raise ValueError('A team has multiple games in a week')
        points, projected, actual = [], 0.0, 0
        for week in range(1, 19):
            game = by_week.get(week)
            point = {'week': week, 'opponent': 'Bye', 'win_probability': None, 'status': 'Bye', 'actual_win': None}
            if game:
                side_now = 'home' if team == game['home_team'] else 'away'
                other = 'away' if side_now == 'home' else 'home'
                p = float(game[side_now + '_win_probability'])
                if not 0 <= p <= 1:
                    raise ValueError('Invalid probability')
                final = game.get('status') == 'Final'
                win = int(game[side_now + '_score'] > game[other + '_score']) if final else None
                actual += win or 0
                projected += win if final else p
                point.update(opponent=game.get(other + '_team_name', game[other + '_team']),
                             win_probability=p, status='Final' if final else 'Scheduled', actual_win=win,
                             home=side_now == 'home', game_id=game['game_id'])
            point.update(projected_wins=round(projected, 6), actual_wins=actual if week <= last_complete and (not game or point['status'] == 'Final') else None)
            points.append(point)
        result.append({'team': team, 'name': first.get(side + '_team_name', team), 'logo': first.get(side + '_logo', ''), 'points': points})
    return {'teams': result, 'note': 'Weekly probabilities retain archived pregame picks. Cumulative projections fix completed wins and sum remaining raw model probabilities; they are not the noisy Monte Carlo estimates. Byes have no win probability.'}


def export(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    data = build_series(json.loads((root / 'public_site/data/predictions.json').read_text()))
    (root / 'public_site/data/season_charts.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    return data


if __name__ == '__main__':
    export()

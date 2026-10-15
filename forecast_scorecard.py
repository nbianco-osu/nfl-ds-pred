"""R49: strict pregame evidence, independently gated by prediction target."""
import json
from pathlib import Path

import pandas as pd
from audit_io import write_json
from performance import summarize


def before(value, cutoff):
    try:
        t, k = pd.Timestamp(value), pd.Timestamp(cutoff)
        return bool(pd.notna(t) and pd.notna(k) and t.tzinfo and k.tzinfo and t < k)
    except (TypeError, ValueError):
        return False


def cohort(frame, kickoffs, target, provenance):
    field = 'winner_forecast_at' if target == 'winner' else 'score_forecast_at'
    eligible = frame.apply(lambda r: before(r.get(field), kickoffs.get(r.game_id)), axis=1)
    selected = frame.loc[eligible if provenance == 'pregame' else ~eligible].copy()
    if target == 'winner':
        selected['predicted_home_score'] = None
        selected['predicted_away_score'] = None
        selected['score_forecast_at'] = None
        selected['spread_grade'] = selected['total_grade'] = 'Not applicable'
    else:
        selected['home_win_probability'] = None
        # A score timestamp alone does not establish a contemporaneous market line.
        line_ok = selected.apply(lambda r: before(r.get('line_observed_at'), kickoffs.get(r.game_id)), axis=1)
        selected.loc[~line_ok, ['spread_grade', 'total_grade']] = 'No pregame line archive'
    return selected


def export(root):
    root = Path(root)
    folder = root / 'public_site/data'
    rows = pd.read_json(folder / 'predictions.json', convert_dates=False)
    kickoffs = json.loads((folder / 'kickoffs.json').read_text())
    groups = []
    for target, version_field in [('winner', 'model_version'), ('score', 'score_model_version')]:
        for provenance in ['pregame', 'unverified']:
            selected = cohort(rows, kickoffs, target, provenance)
            for version, part in selected.groupby(selected[version_field].fillna('legacy')):
                groups.append({'target': target, 'provenance': provenance, 'version': version,
                    'summary': summarize(part), 'weeks': [
                        {'season': int(s), 'week': int(w), **summarize(g)}
                        for (s, w), g in part.groupby(['season', 'week']) if g.status.eq('Final').any()]})
    report = {'schema_version': 1, 'groups': groups,
        'note': 'Pregame requires a timezone-aware saved forecast timestamp strictly before the scheduled kickoff. Winner and score cohorts are independent; line grades additionally require a pregame line observation. Untimestamped or late records are unverified, not prospective evidence. Retrospective research remains in Evidence Lab. Ties are excluded from winner metrics; pushes and ungraded lines are excluded from hit rates. No mixed-version default.'}
    write_json(folder / 'forecast_scorecard.json', report)
    return report


if __name__ == '__main__':
    export(Path(__file__).resolve().parent)

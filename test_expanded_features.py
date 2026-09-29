import unittest
import numpy as np
import pandas as pd

from expanded_features import pregame_features, aggregate_ngs
from season_charts import build_series


class ExpandedFeatureTests(unittest.TestCase):
    def test_no_same_week_scores_or_stats(self):
        schedule = pd.DataFrame([dict(game_id=f'g{i}', season=2026, week=i, home_team='SEA', away_team='NE', home_score=20, away_score=10) for i in [1, 2, 3]])
        stats = pd.DataFrame([dict(game_id=f'g{i}', team=t, epa=i * .1) for i in [1, 2, 3] for t in ['SEA', 'NE']])
        initial = pregame_features(schedule, stats)
        stats.loc[stats.game_id.eq('g2'), 'epa'] = 99
        schedule.loc[schedule.week.eq(2), ['home_score', 'away_score']] = [0, 50]
        changed = pregame_features(schedule, stats)
        pd.testing.assert_frame_equal(initial.iloc[:2], changed.iloc[:2])
        self.assertNotEqual(initial.iloc[2].home_team_ext_epa_last5, changed.iloc[2].home_team_ext_epa_last5)
        self.assertNotEqual(initial.iloc[2].home_team_ext_elo, changed.iloc[2].home_team_ext_elo)

    def test_ngs_weighting_and_no_season_summary(self):
        frame = pd.DataFrame([dict(season=2026, season_type='REG', week=w, team_abbr='NE', targets=t,
                                   avg_separation=s, avg_yac_above_expectation=s) for w, t, s in [(1, 2, 1), (1, 6, 3), (0, 100, 99)]])
        result = aggregate_ngs(frame, 'receiving')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0].ngs_avg_separation, 2.5)

    def test_series_byes_ties_and_conservation(self):
        rows = [dict(game_id='g1', week=1, home_team='SEA', away_team='NE', home_win_probability=.6,
                     away_win_probability=.4, status='Final', home_score=10, away_score=10),
                dict(game_id='g3', week=3, home_team='NE', away_team='SEA', home_win_probability=.7,
                     away_win_probability=.3, status='Scheduled')]
        result = build_series(rows)
        sea = next(t for t in result['teams'] if t['team'] == 'SEA')['points']
        self.assertIsNone(sea[1]['win_probability'])
        self.assertEqual(sea[0]['projected_wins'], 0)
        self.assertEqual(sea[1]['projected_wins'], 0)
        self.assertEqual(sea[2]['projected_wins'], .3)
        self.assertIsNone(sea[2]['actual_wins'])
        self.assertEqual(sum(t['points'][-1]['projected_wins'] for t in result['teams']), 1)


if __name__ == '__main__':
    unittest.main()

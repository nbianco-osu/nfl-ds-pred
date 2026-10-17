import unittest
import numpy as np
import pandas as pd
from robust_ensemble import RobustMatchupEnsemble, fit_weights
from train_extension240 import recipes
from train_robust_ensemble import separated
from predict_robust import validate_games
from robust_features import dependency_columns
from predict_schedule import build_schedule_features


class Member:
    def predict_proba(self, x):
        return np.tile([.4, .6], (len(x), 1))

    def predict(self, x):
        return np.tile([24., 20.], (len(x), 1))


class EnsembleTests(unittest.TestCase):
    def model(self):
        return RobustMatchupEnsemble([Member()]*3, [1/3]*3, [Member()]*3,
            [1/3]*3, Member(), [[-2, -1], [2, 1]], ['a', 'b'], 'test')

    def test_recipe_count_and_unique(self):
        rows = list(recipes({'core_updated': ['a'], 'charted_updated': ['a', 'b']}))
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({r[0] for r in rows}), 120)
        self.assertEqual(sum(r[1] == 'winner' for r in rows), 60)
        self.assertEqual(sum(r[1] == 'score' for r in rows), 60)

    def test_network_no_random_validation(self):
        for name, _, estimator, _ in recipes({'core_updated': ['a']}):
            if '_mlp_' in name:
                params = estimator.get_params(deep=True)
                self.assertFalse(params.get('early_stopping', params.get('regressor__early_stopping')))

    def test_weight_limits_and_determinism(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(.1, .9, (100, 6)); y = (p[:, 0] > .5).astype(int)
        w = fit_weights(p, y, True)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertLessEqual(w.max(), .3500001)
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w, fit_weights(p, y, True))

    def test_score_weights(self):
        rng = np.random.default_rng(42)
        x = rng.normal(23, 5, (40, 2, 6)); y = rng.normal(23, 5, (40, 2))
        self.assertAlmostEqual(fit_weights(x, y).sum(), 1)

    def test_reject_invalid_weights_input(self):
        for p in [np.full((10, 6), np.nan), np.full((10, 6), 2)]:
            with self.assertRaises(ValueError): fit_weights(p, np.zeros(10), True)
        with self.assertRaises(ValueError): fit_weights(np.ones((10, 2)), np.zeros(10))

    def test_inference_contract_and_ranges(self):
        x = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        model = self.model()
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
        np.testing.assert_allclose(model.predict(x), [[24, 20]]*2)
        details = model.predict_details(x)
        self.assertTrue((details['score_lower'] <= details['score_upper']).all())
        self.assertTrue((details['score_lower'] >= 0).all())
        self.assertTrue((details['score_implied_home_win'] == 1).all())

    def test_fail_closed_bad_input(self):
        for x in [pd.DataFrame({'a': [1]}), pd.DataFrame({'a': [np.inf], 'b': [1]}),
                  pd.DataFrame({'a': [np.nan], 'b': [1]}),
                  pd.DataFrame([[1, 2]], columns=['a', 'a'])]:
            with self.assertRaises(ValueError): self.model().predict(x)

    def test_invalid_residuals_and_member_weights(self):
        with self.assertRaises(ValueError):
            RobustMatchupEnsemble([Member()], [1], [Member()], [1], Member(), [], ['a'], 'x')
        with self.assertRaises(ValueError):
            RobustMatchupEnsemble([Member()], [2], [Member()], [1], Member(), [[1, 2]], ['a'], 'x')

    def test_separation_dates_and_ids(self):
        a = pd.DataFrame({'game_id': ['a'], 'gameday': pd.to_datetime(['2024-01-01'])})
        b = pd.DataFrame({'game_id': ['b'], 'gameday': pd.to_datetime(['2025-01-01'])})
        separated(a, b)
        with self.assertRaises(ValueError): separated(b, a)
        with self.assertRaises(ValueError): separated(a, a)
        with self.assertRaises(ValueError): separated(a, b.assign(game_id='a'))

    def test_future_only_inference(self):
        frame = pd.DataFrame({'game_id':['future'], 'kickoff_at':['2026-10-18T17:00:00Z']})
        validate_games(frame, '2026-10-12', '2026-10-17T12:00:00Z')
        with self.assertRaises(ValueError): validate_games(frame, '2026-10-12', '2026-10-19T12:00:00Z')
        with self.assertRaises(ValueError): validate_games(frame, '2026-10-18', '2026-10-17T12:00:00Z')
        with self.assertRaises(ValueError): validate_games(frame.assign(kickoff_at='2026-10-18'), '2026-10-12', '2026-10-17T12:00:00Z')
        with self.assertRaises(ValueError): validate_games(pd.concat([frame, frame]), '2026-10-12', '2026-10-17T12:00:00Z')

    def test_live_differentials_keep_side_dependencies(self):
        history = pd.DataFrame([dict(game_id='prior', season=2026, week=1,
            home_team='H', away_team='A', home_score=24, away_score=17)])
        upcoming = pd.DataFrame([dict(game_id='future', season=2026, week=2,
            home_team='H', away_team='A')])
        cols = ['diff_team_win_pct', 'diff_team_pbp_off_epa_per_play_avg']
        snapshots = pd.DataFrame([dict(game_id='future', home_team_pbp_off_epa_per_play_avg=.2,
            away_team_pbp_off_epa_per_play_avg=-.1)]).set_index('game_id')
        _, broken = build_schedule_features(upcoming, history, {'feature_cols':cols}, snapshots)
        self.assertTrue(broken[cols].isna().all().all())
        _, fixed = build_schedule_features(upcoming, history, {'feature_cols':dependency_columns(cols)}, snapshots)
        self.assertEqual(fixed.at[0, cols[0]], 1)
        self.assertAlmostEqual(fixed.at[0, cols[1]], .3)


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
import pandas as pd
from opponent_ratings import build, fit_ratings, prepare


class OpponentRatingsTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([
            dict(game_id=f'g{w}', season=2026, week=w, gameday=f'2026-09-{w*7:02}',
                 game_type='REG', home_team='A', away_team='B', home_score=24 if w<3 else np.nan,
                 away_score=17 if w<3 else np.nan) for w in [1, 2, 3]])
        self.pbp = pd.DataFrame([
            dict(game_id=f'g{w}', play_id=i, posteam=t, defteam='B' if t=='A' else 'A',
                 play_type='pass', epa=.5 if t=='A' else -.2, sack=int(t=='B'),
                 pass_attempt=1, rush_attempt=0, yards_gained=20 if t=='A' else -5,
                 qb_dropback=1) for w in [1,2] for i,t in enumerate(['A','B'])])

    def test_temporal_boundary(self):
        before, _ = build(self.pbp, self.games, 2026)
        changed = self.pbp.copy()
        changed.loc[changed.game_id.eq('g2'), 'epa'] = 100
        after, _ = build(changed, self.games, 2026)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
        self.assertTrue(before.loc[0, 'home_net_epa'] is None or pd.isna(before.loc[0, 'home_net_epa']))
        self.assertNotEqual(before.loc[2,'home_net_epa'], after.loc[2,'home_net_epa'])

    def test_signs_and_differentials(self):
        ratings, _ = fit_ratings(prepare(self.pbp,self.games), ['A','B','C'])
        a,b,c = ratings
        self.assertGreater(a['net_epa'], b['net_epa'])
        self.assertEqual(a['sack_differential_per_game'], 1)
        self.assertEqual(a['big_play_differential_per_game'], 1)
        self.assertEqual(a['sack_rate_differential'], 1)
        self.assertIsNone(c['net_epa'])
        self.assertAlmostEqual(sum(r['net_epa'] for r in [a,b]), 0)

    def test_shrinkage(self):
        p = prepare(self.pbp,self.games)
        low,_ = fit_ratings(p,['A','B'],alpha=1)
        high,_ = fit_ratings(p,['A','B'],alpha=1000)
        self.assertLess(abs(high[0]['net_epa']),abs(low[0]['net_epa']))

    def test_missing_not_zero_and_duplicates_rejected(self):
        p = self.pbp.copy()
        p.loc[0,'sack'] = np.nan
        rows,_ = fit_ratings(prepare(p,self.games),['A','B'])
        self.assertIsNone(rows[0]['sack_differential_per_game'])
        with self.assertRaises(ValueError):
            prepare(pd.concat([self.pbp,self.pbp]), self.games)


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from generative_overtime import resolve,finalize
from generative_season import simulate as season_simulate
from generative_pipeline import weights,features,probs
from generative_tracking import eligible,save_once,review
from markov_research import simulate_batch,fit
from test_markov_research import drives


class OvertimeTests(unittest.TestCase):
    def test_opening_td_allows_reply(self):
        np.testing.assert_array_equal(resolve([(120,7,0),(120,7,0),(100,3,0)]),[10,7])
    def test_opening_fg_reply_td_has_no_extra_point(self):
        np.testing.assert_array_equal(resolve([(120,3,0),(120,7,0)]),[3,6])
    def test_punt_then_td_is_sudden_death(self):
        np.testing.assert_array_equal(resolve([(120,0,0),(120,8,0)]),[0,6])
    def test_defensive_td_and_safety_end_game(self):
        np.testing.assert_array_equal(resolve([(120,0,7)]),[0,6])
        np.testing.assert_array_equal(resolve([(120,0,2)]),[0,2])
    def test_clock_can_expire_before_reply(self):
        np.testing.assert_array_equal(resolve([(590,3,0),(20,7,0)]),[3,0])
        np.testing.assert_array_equal(resolve([(601,7,0)]),[0,0])
    def test_reply_td_may_need_conversion(self):
        np.testing.assert_array_equal(resolve([(120,7,0),(120,8,0)]),[7,8])
    def test_regulation_non_ties_unchanged(self):
        paths=np.tile([[[0,0],[7,3]]],(10,1,1))
        np.testing.assert_array_equal(finalize(paths,fit(drives(),2026,2),'A','B'),paths[:,-1])


class GenerativeTests(unittest.TestCase):
    def test_batch_paths_clock_and_determinism(self):
        model=fit(drives(),2026,2)
        a=simulate_batch(model,'A','B',100)
        np.testing.assert_array_equal(a,simulate_batch(model,'A','B',100))
        self.assertTrue((np.diff(a,axis=1)>=0).all())
        self.assertAlmostEqual(probs(a[:,-1]).sum(),1)
        self.assertFalse(simulate_batch(fit(drives().assign(duration=2000),2026,2),'A','B',10).any())
    def test_weights_simplex_and_fallback(self):
        np.testing.assert_allclose(weights([]),[1/3]*3)
        p=[dict(probabilities=[[.9,.05,.05],[.1,.1,.8],[.1,.1,.8]],outcome=0)]*20
        w=weights(p);self.assertAlmostEqual(w.sum(),1);self.assertGreater(w[0],.8)
    def test_feature_cutoff_ignores_current_week(self):
        history=pd.DataFrame([dict(game_id='a',season=2026,week=w,gameday='2026-10-01',home_team='A',away_team='B',home_score=s,away_score=10) for w,s in [(1,20),(2,99)]])
        games=history.iloc[[1]]
        self.assertEqual(features(history,games).iloc[0].home_for,20)
    def test_season_fixed_results_and_conservation(self):
        s=pd.DataFrame([dict(game_id='a',week=1,home_team='A',away_team='B',home_score=7,away_score=3),dict(game_id='b',week=2,home_team='B',away_team='A',home_score=np.nan,away_score=np.nan)])
        r=season_simulate(s,{'b':dict(home_win=0,tie=1,away_win=0)},trials=100)
        self.assertTrue(r['conservation_passed'])
        self.assertEqual(r['teams']['A'][0]['low'],1)
        self.assertEqual(r['teams']['A'][-1]['mean_wins'],1)
        self.assertEqual(r['teams']['B'][-1]['mean_ties'],1)
        with self.assertRaises(ValueError):season_simulate(s,{})
    def test_real_window_and_training_cutoff(self):
        g=dict(season=2026,week=7,gameday='2026-10-25',gametime='13:00',home_score=None,away_score=None)
        a=dict(season=2026,training_max_week=6)
        self.assertTrue(eligible(g,a,'2026-10-25T15:30:00Z','90m'))
        self.assertFalse(eligible(g,a,'2026-10-25T17:01:00Z','90m'))
        self.assertFalse(eligible(g,{**a,'training_max_week':7},'2026-10-25T15:30:00Z','90m'))
    def test_immutable_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json'
            self.assertTrue(save_once(p,{'a':1}));self.assertFalse(save_once(p,{'a':2}))
            self.assertEqual(json.loads(p.read_text()),{'a':1})

    def test_ties_excluded_and_repeated_review_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'generative-test'
            record=dict(model_version=folder.name,game_id='g',horizon='90m',forecast_at='2026-10-25T15:30:00Z',
                        prediction={'distributions':{'ensemble':{'home_win':.4,'tie':.1,'away_win':.5}}})
            save_once(folder/'forecasts/g_90m.json',record)
            schedule=pd.DataFrame([dict(game_id='g',week=7,gameday='2026-10-25',gametime='13:00',home_score=20,away_score=20)])
            a=review(folder,schedule);b=review(folder,schedule)
            self.assertEqual(a,b);self.assertEqual(a['matched_games'],1);self.assertEqual(a['decisive_games'],0)
            self.assertIsNone(a['accuracy']);self.assertFalse(a['ready_for_review'])
            schedule['home_score']=21
            self.assertEqual(review(folder,schedule)['result_corrections'],['g'])

    def test_migration_scope_is_exact(self):
        from runner.migrate_generative import allowed
        files={f'models/generative/generat ive/{name}' for name in ['candidate.joblib','report.json']}
        files.add('public_site/data/generative.json')
        self.assertTrue(allowed(files,'generat ive'))
        self.assertFalse(allowed(files|{'models/production.joblib'},'generat ive'))


if __name__=='__main__':unittest.main()

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from audit_io import write_json
from next_evidence import weights, radius
from shadow_next import append_forecast, grade, horizon, safe_version, paired_production
from research_dataset import lagged
from pregame_refresh import due
from train_extension120 import recipes
from refresh_lock import refresh_lock


class NextReleaseTests(unittest.TestCase):
    def game(self):
        return dict(game_id='2026_07_A_B',season=2026,week=7,game_type='REG',gameday='2026-10-18',gametime='13:00',home_team='B',away_team='A',home_score=np.nan,away_score=np.nan)

    def record(self,task='winner'):
        return dict(generation='test',model='model',game_id='2026_07_A_B',horizon='initial',task=task,
                    forecast_at='2026-10-16T17:00:00+00:00',kickoff_at_observation='2026-10-18T17:00:00+00:00',
                    training_through='2026-10-12',home_probability=.7)

    def test_exactly_sixty_additions(self):
        r=list(recipes({'core_updated':['a'],'charted_updated':['a','b']}))
        self.assertEqual(len(r),60);self.assertEqual(len({x[0] for x in r}),60)
        self.assertEqual(sum(x[0].startswith('nn_') for x in r),48)

    def test_first_forecast_immutable(self):
        saved={};r=self.record();now=pd.Timestamp(r['forecast_at'])
        key=append_forecast(saved,r,self.game(),now)
        append_forecast(saved,dict(r,home_probability=.1),self.game(),now)
        self.assertEqual(saved[key]['home_probability'],.7)

    def test_late_and_training_overlap_rejected(self):
        r=self.record()
        with self.assertRaises(ValueError):append_forecast({},r,self.game(),pd.Timestamp('2026-10-18T18:00:00Z'))
        r['training_through']='2026-10-18'
        with self.assertRaises(ValueError):append_forecast({},r,self.game(),pd.Timestamp(r['forecast_at']))

    def test_ties_and_model_versions_separated(self):
        g=self.game();g.update(home_score=20,away_score=20)
        r=self.record();other=dict(r,generation='next')
        result=grade({'a':r,'b':other},pd.DataFrame([g]))
        self.assertEqual(len(result),2)
        self.assertTrue(all(x['graded']==0 and x['ties_excluded']==1 for x in result))

    def test_score_push_and_missing_lines(self):
        g=self.game();g.update(home_score=24,away_score=21)
        r=self.record('score');r.update(predicted_home_score=25,predicted_away_score=20,market_home_spread=-3,market_total=45)
        result=grade({'a':r},pd.DataFrame([g]))[0]
        self.assertEqual(result['spread_pushes'],1)
        self.assertEqual(result['total_ungraded'],1)  # projected total exactly on line: no edge
        r.update(market_home_spread=None,market_total=None)
        self.assertEqual(grade({'a':r},pd.DataFrame([g]))[0]['spread_ungraded'],1)

    def test_window_and_duplicate_guard(self):
        now=pd.Timestamp('2026-10-17T17:00:00Z');g=self.game()
        self.assertEqual(horizon(g,now),'24h')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();pd.DataFrame([g]).to_csv(root/'data/schedule_2026.csv',index=False)
            self.assertEqual(len(due(root,2026,now)),1)
            write_json(root/'audit/checkpoints.json',{g['game_id']+'|24h':{}})
            self.assertEqual(due(root,2026,now),[])

    def test_convex_weights_and_radius(self):
        w=weights(np.array([[.1,.9],[.9,.1]]),np.array([0,1]),True)
        self.assertAlmostEqual(w.sum(),1);self.assertGreater(w[0],.99)
        self.assertEqual(radius([1,2,3,4],.8),4)
        with self.assertRaises(ValueError):radius([np.nan])

    def test_lock_released(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with refresh_lock(root):
                with self.assertRaises(RuntimeError):
                    with refresh_lock(root):pass
            with refresh_lock(root):pass

    def test_generation_path(self):
        with self.assertRaises(ValueError):safe_version('../outside')

    def test_prospective_pairing(self):
        g=self.game();g.update(home_score=24,away_score=21)
        r=self.record();ref=dict(r,model='production_v1',home_probability=.6)
        rows=paired_production({'a':r,'b':ref},pd.DataFrame([g]))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['games'],1)
        self.assertLess(rows[0]['delta_log_loss'],0);self.assertIsNone(rows[0]['ci95'])
        ref['generation']='other'
        self.assertEqual(paired_production({'a':r,'b':ref},pd.DataFrame([g])),[])

    def test_lagged_target_filter_preserves_history(self):
        a=self.game();a.update(week=6,home_score=20,away_score=10,game_id='prior')
        b=self.game()
        schedule=pd.DataFrame([a,b]);stats=pd.DataFrame([{'game_id':'prior','team':'A','metric':2},{'game_id':'prior','team':'B','metric':3}])
        full=lagged(schedule,stats).set_index('game_id')
        selected=lagged(schedule,stats,[b['game_id']]).set_index('game_id')
        pd.testing.assert_frame_equal(full.loc[[b['game_id']]],selected)


if __name__=='__main__':unittest.main()

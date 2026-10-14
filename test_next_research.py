import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pregame_inputs import import_rows, select, timestamp, validate
from score_experiments import sample, crps, evaluate
from temporal_stack import build
from warehouse_queue import snapshots, query


class ScoreResearchTests(unittest.TestCase):
    def test_draws_finite_reproducible_and_means(self):
        for kind in ['poisson','tweedie','scoring_events','ridge_residual']:
            artifact={'kind':kind,'dispersion':np.array([1.,1.]),'residuals':np.zeros((128,2))}
            a=sample(artifact,[24,20]);b=sample(artifact,[24,20])
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.isfinite(a).all() and (a>=0).all())
            np.testing.assert_allclose(a.mean(axis=0),[24,20],atol=.7)

    def test_crps_and_push(self):
        self.assertEqual(crps([1,1,1],1),0)
        self.assertEqual(crps([1,1,1],3),2)
        result=evaluate(np.array([[24,20]]*100),24,20,4,44)
        self.assertEqual(result['total_crps'],0)
        self.assertLess(result['margin_brier3'],.001)
        self.assertEqual(result['total_coverage80'],1)

    def test_shared_report_cohort(self):
        root=Path(__file__).resolve().parent
        rows=pd.read_csv(root/'backtests/score_experiments.csv')
        self.assertFalse(rows.duplicated(['model','game_id']).any())
        cohorts=[set(g.game_id) for _,g in rows.groupby('model')]
        self.assertEqual(len(cohorts),4)
        self.assertTrue(all(c==cohorts[0] for c in cohorts))


class InputArchiveTests(unittest.TestCase):
    def setUp(self):
        self.game={'game_id':'2026_06_SEA_SF','season':2026,'week':6,'gameday':'2026-10-18',
                   'gametime':'16:25','home_team':'SF','away_team':'SEA','stadium_id':'SFO01',
                   'home_score':np.nan,'away_score':np.nan}
        self.row={'kind':'starter','game_id':self.game['game_id'],'team':'SEA','status':'confirmed',
                  'player_id':'test-player','player_name':'Test Player','provider':'Test supplied input',
                  'source_url':'https://example.com/announcement','published_at':'2026-10-14T10:00:00Z'}
        self.now=timestamp('2026-10-14T12:00:00Z')

    def test_late_future_unconfirmed_and_timezone_rejected(self):
        for changed,now in [({},'2026-10-19T00:00:00Z'),({'published_at':'2026-10-15T00:00:00Z'},self.now),
                            ({'status':'probable'},self.now),({'published_at':'2026-10-14'},self.now),
                            ({'player_id':np.nan},self.now)]:
            with self.assertRaises(ValueError):validate(self.row|changed,self.game,timestamp(now))

    def test_atomic_batch_and_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir()
            pd.DataFrame([self.game]).to_csv(root/'data/schedule_2026.csv',index=False)
            with self.assertRaises(ValueError):
                import_rows(root,[self.row,self.row|{'status':'expected'}],2026,self.now)
            self.assertFalse((root/'audit/pregame_inputs.json').exists())
            self.assertEqual(import_rows(root,[self.row],2026,self.now),1)
            self.assertEqual(import_rows(root,[self.row],2026,self.now+pd.Timedelta(hours=1)),0)
            records=list(json.loads((root/'audit/pregame_inputs.json').read_text())['observations'].values())
            self.assertEqual(select(records,self.game,self.now-pd.Timedelta(seconds=1)),{})
            self.assertEqual(len(select(records,self.game,self.now)),1)

    def test_weather_stadium_and_validity(self):
        row={k:v for k,v in self.row.items() if k not in ['team','player_id','player_name','status']}
        row.update(kind='weather',stadium_id='SFO01',temperature_f=65,wind_mph=8,valid_at='2026-10-18T20:00:00Z')
        validate(row,self.game,self.now)
        for change in [{'stadium_id':'OTHER'},{'wind_mph':-1},{'valid_at':'2026-10-19T20:00:00Z'}]:
            with self.assertRaises(ValueError):validate(row|change,self.game,self.now)


class StackResearchTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for year in [2024,2025,2026]:
            for i in range(12):
                for model in ['one','two']:
                    rows.append({'game_id':f'{year}_{i}','season':year,'week':i+1,'gameday':f'{year}-10-{i+1:02}',
                                 'model_id':model,'home_win':i%2,'probability':.4+.2*(i%2)})
        return pd.DataFrame(rows),{'models':[{'id':m,'market':False} for m in ['one','two']],
                                  'folds':[{'final_calibration_through':'2023-12-31','test_start':'2024-09-01'}]}

    def test_outer_labels_cannot_change_forecasts(self):
        rows,report=self.fixture();_,_,a=build(rows,report)
        rows.loc[rows.season.eq(2026),'home_win']=1-rows.loc[rows.season.eq(2026),'home_win']
        _,_,b=build(rows,report)
        np.testing.assert_allclose(a.probability,b.probability)

    def test_missing_base_and_label_disagreement_rejected(self):
        rows,report=self.fixture()
        with self.assertRaises(ValueError):build(rows.iloc[1:],report)
        rows.loc[0,'home_win']=1
        with self.assertRaises(ValueError):build(rows,report)


class QueueTests(unittest.TestCase):
    def test_insert_only_and_full_payload_verification(self):
        sql=query([{'id':'x','payload':{'name':"O'Brien"}}])
        self.assertIn("O''Brien",sql)
        self.assertIn('on conflict(id) do nothing',sql)
        self.assertIn("(d.payload-'updated_at')=i.payload",sql)
        self.assertNotIn('delete from',sql)

    def test_operational_timestamps_do_not_change_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir();(root/'public_site/data').mkdir(parents=True)
            pd.DataFrame([{'game_id':'g','season':2026}]).to_csv(root/'data/schedule_2026.csv',index=False)
            pd.DataFrame([{'team_abbr':'SEA'}]).to_csv(root/'data/team_metadata.csv',index=False)
            path=root/'public_site/data/predictions.json'
            path.write_text(json.dumps([{'game_id':'g','season':2026,'updated_at':'old'}]))
            before=list(snapshots(root,2026))
            path.write_text(json.dumps([{'game_id':'g','season':2026,'updated_at':'new'}]))
            self.assertEqual(before,list(snapshots(root,2026)))

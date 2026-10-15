import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from live_challengers import sample, form_features, export, grade
from weather_inputs import observation
from pregame_inputs import select
from reliability import export as monitor


class FixedScores:
    def predict(self,x):
        return np.tile([24.,20.],(len(x),1))


class Release23Tests(unittest.TestCase):
    def game(self):
        return {'game_id':'g','season':2026,'game_type':'REG','week':6,'gameday':'2026-10-18',
                'gametime':'13:00','stadium_id':'S','home_team':'A','away_team':'B',
                'home_score':np.nan,'away_score':np.nan,'roof':'outdoors'}

    def test_weather_observation_time_not_publication(self):
        payload={'hourly_units':{'temperature_2m':'F','wind_speed_10m':'mp/h'},
                 'hourly':{'time':['2026-10-18T17:00'],'temperature_2m':[60],'wind_speed_10m':[8]}}
        row=observation(self.game(),payload,'2026-10-15T12:00:00Z','https://open-meteo.com','hash')
        self.assertIsNone(row['published_at'])
        self.assertEqual(select([row],self.game(),'2026-10-14T12:00Z'),{})
        self.assertIn('weather',select([row],self.game(),'2026-10-16T12:00Z'))
        with self.assertRaises(ValueError):
            observation(self.game(),payload,'2026-10-19T12:00Z','https://open-meteo.com','hash')
        payload['hourly']['time']=['2026-10-17T17:00']
        with self.assertRaises(ValueError):
            observation(self.game(),payload,'2026-10-15T12:00Z','https://open-meteo.com','hash')

    def test_overdispersion_and_correlation(self):
        artifact={'dispersion':[.15,.15],'rho':.5}
        poisson=sample(artifact,[24,20],'poisson')
        correlated=sample(artifact,[24,20],'correlated_negative_binomial')
        np.testing.assert_array_equal(correlated,sample(artifact,[24,20],'correlated_negative_binomial'))
        self.assertTrue(np.isfinite(correlated).all())
        self.assertTrue((correlated>=0).all())
        self.assertGreater(correlated[:,0].var(),2*poisson[:,0].var())
        self.assertGreater(np.corrcoef(correlated.T)[0,1],.25)
        np.testing.assert_allclose(correlated.mean(axis=0),[24,20],atol=.5)

    def test_future_and_same_week_results_excluded(self):
        g=self.game(); schedule=pd.DataFrame([g])
        data=pd.DataFrame([dict(g,game_id='prior',gameday=pd.Timestamp('2026-10-01'),home_score=21,away_score=14),
                           dict(g,gameday=pd.Timestamp(g['gameday']),home_score=99,away_score=0)])
        x,_=form_features(g,schedule,data)
        self.assertEqual(x.iloc[0].tolist(),[21,14,14,21])

    def test_archive_immutable_and_ties_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for p in ['data','public_site/data','models/research']:(root/p).mkdir(parents=True,exist_ok=True)
            g=self.game(); schedule=pd.DataFrame([g]);schedule.to_csv(root/'data/schedule_2026.csv',index=False)
            (root/'public_site/data/predictions.json').write_text(json.dumps([{'game_id':'g','home_win_probability':.6,'model_version':'prod'}]))
            artifact={'model':FixedScores(),'model_version':'test','calibration_through':'2026-01-04',
                      'fit_through':'2025-01-05','dispersion':[.1,.1],'rho':0}
            joblib.dump(artifact,root/'models/research/test.joblib')
            (root/'models/research/live_challengers.json').write_text(json.dumps({'season':2026,'artifact':'models/research/test.joblib'}))
            past=pd.DataFrame([dict(g,gameday=pd.Timestamp('2026-09-01'),home_score=20,away_score=17)])
            with patch('live_challengers.history',return_value=(past,{})),patch('live_challengers.sample',return_value=np.array([[24,20],[20,24],[21,21]])):
                export(root,now='2026-10-15T12:00Z')
                path=root/'learning/live_challengers/test/forecasts.json';before=path.read_bytes()
                export(root,now='2026-10-16T12:00Z')
                self.assertEqual(before,path.read_bytes())
                schedule.loc[0,['home_score','away_score']]=21
                schedule.to_csv(root/'data/schedule_2026.csv',index=False)
                export(root,now='2026-10-19T12:00Z')
                self.assertEqual(before,path.read_bytes())
                self.assertTrue(all(r['games']==0 for r in grade(list(json.loads(before).values()),schedule)))

    def test_monitor_stale_and_missing_without_raw_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            pd.DataFrame([self.game()]).to_csv(root/'data/schedule_2026.csv',index=False)
            r=monitor(root,now='2026-10-15T12:00Z')
            ids={x['id'] for x in r['issues']}
            self.assertTrue({'stale_refresh','missing_predictions','missing_starters','missing_weather','missing_challengers'}<=ids)
            self.assertEqual(r['issue_fingerprint'],monitor(root,now='2026-10-15T13:00Z')['issue_fingerprint'])


if __name__=='__main__':unittest.main()

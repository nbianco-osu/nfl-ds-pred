import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from audit_io import write_json
from iteration_r61 import guarded_inference,prune_oof,review,vintage
from possession_research import crps,simulate,parameters
from runner.migrate_candidates import allowed_additions


class IterationTests(unittest.TestCase):
    def test_public_research_allowlist(self):
        from runner.run import valid_public_export
        self.assertTrue(valid_public_export('backtests/retraining/expanded240-test/report.json'))
        for name in ['backtests/retraining/x/model.joblib', 'backtests/retraining/../secret.json',
                     'models/retrained/x/report.json', 'backtests/private.json']:
            self.assertFalse(valid_public_export(name))

    def test_migration_scope(self):
        self.assertTrue(allowed_additions({'models/retrained/new/a.joblib':'x'},'new','ens'))
        self.assertFalse(allowed_additions({'models/production.joblib':'x'},'new','ens'))
        self.assertFalse(allowed_additions({'audit/checkpoints.json':'x'},'new','ens'))
        self.assertFalse(allowed_additions({},'new','ens'))

    def test_fallback_requires_matching_approval(self):
        candidate=lambda:12;champion=lambda:9
        self.assertEqual(guarded_inference(candidate,champion,'sha')['forecast'],9)
        approval=dict(approved=True,artifact_sha256='sha',reviewer='test',approved_at='2026-01-01')
        self.assertEqual(guarded_inference(candidate,champion,'sha',approval)['route'],'candidate')
        def failed():raise ValueError('private details')
        result=guarded_inference(failed,champion,'sha',approval)
        self.assertEqual(result['forecast'],9);self.assertNotIn('private details',str(result))
        self.assertEqual(guarded_inference(candidate,champion,'other',approval)['route'],'champion')

    def test_crps_exact_small_sample(self):
        self.assertEqual(crps([0,2],1),.5)
        with self.assertRaises(ValueError):crps([],1)

    def test_possession_outputs(self):
        params=dict(rates=np.array([.15,.2,.01,.001]),td_points=np.array([.05,.9,.05]),drives=11)
        a=simulate([24,20],params,trials=200)
        np.testing.assert_array_equal(a,simulate([24,20],params,trials=200))
        self.assertEqual(a.shape,(200,2));self.assertTrue((a>=0).all());self.assertTrue(np.issubdtype(a.dtype,np.integer))
        with self.assertRaises(ValueError):simulate([-1,20],params)

    def test_insufficient_drives(self):
        frame=pd.DataFrame([dict(game_id='one',posteam='A',fixed_drive=1,fixed_drive_result='Punt')])
        with self.assertRaises(ValueError):parameters(frame)

    def test_pruning_no_test_labels(self):
        records=[]
        for name in ['r59_winner_a','r59_winner_b','r59_winner_c']:
            for i in range(10):records.append(dict(model=name,game_id=str(i),season=2023,
                home_score=24 if i%2 else 10,away_score=20,home_probability=.6 if i%2 else .4))
        frame=pd.DataFrame(records);result=prune_oof(frame,'winner')
        self.assertEqual(len(result['selected']),1)
        with self.assertRaises(ValueError):prune_oof(frame.assign(season=2026),'winner')
        with self.assertRaises(ValueError):prune_oof(pd.concat([frame,frame]),'winner')

    def test_fixed_review_never_promotes_and_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);version='new'
            policy=dict(candidate='ensemble_winner',horizon='90m',review_weeks=[10],minimum_games=[50],
                        minimum_weeks=6,familywise_alpha=.05,looks=3)
            write_json(root/'models/retrained'/version/'review_policy.json',policy)
            write_json(root/'audit/shadow_next'/version/'forecasts.json',{})
            schedule=pd.DataFrame([dict(game_id='g',season=2026,week=10,game_type='REG',home_score=np.nan,away_score=np.nan)])
            self.assertEqual(review(root,version,schedule)['looks'],[])
            schedule[['home_score','away_score']]=[20,10]
            result=review(root,version,schedule)
            self.assertFalse(result['ready_for_review']);self.assertEqual(result['looks'][0]['status'],'Insufficient matched evidence')
            self.assertFalse(result['looks'][0]['automatic_promotion'])
            saved=(root/'audit/fixed_reviews'/version/'week_10.json').read_bytes()
            review(root,version,schedule)
            self.assertEqual(saved,(root/'audit/fixed_reviews'/version/'week_10.json').read_bytes())

    def test_vintage_adds_games_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            schedule=pd.DataFrame([dict(game_id='new',game_type='REG',season=2026,week=6,gameday='2026-10-15',home_score=24,away_score=20)])
            schedule.to_csv(root/'data/schedule_2026.csv',index=False)
            for name in ['current_features_2026.csv','opponent_features_2026.csv']:(root/'data'/name).write_text('game_id\nnew\n')
            write_json(root/'audit/sources/load_pbp/latest.json',{'sha256':'abc'})
            old=pd.DataFrame([dict(game_id='old',season=2025,week=1,gameday=pd.Timestamp('2025-09-01'),home_score=10,away_score=7,f=1)])
            features=pd.DataFrame({'f':[2]},index=pd.Index(['new'],name='game_id'))
            with patch('iteration_r61.load_data',return_value=(old,{'source':'hash'})),patch('iteration_r61.build_live_features',return_value=features):
                report,table=vintage(root,2026,['f']);second,_=vintage(root,2026,['f'])
            self.assertEqual(report,second);self.assertEqual(report['new_completed_games'],['new'])
            self.assertEqual(report['missing_completed_games'],[]);self.assertEqual(len(table),2)


if __name__=='__main__':unittest.main()

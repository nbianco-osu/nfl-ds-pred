import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_io import atomic_bytes, write_json
from active_learning import update_learning
from advanced_backtest import wilson
from challenger_review import report
from current_features import roll_forward
from performance import export
from publication_changes import canonical
from score_distribution import draws, summarize
from staged_refresh import refresh
from validate_dashboard import validate


class IterationSafetyTests(unittest.TestCase):
    def test_timestamp_only_publication_is_suppressed(self):
        self.assertEqual(canonical({'updated_at':'a','health':{'last_successful_refresh':'b'},'probability':.5}),
                         canonical({'updated_at':'c','health':{'last_successful_refresh':'d'},'probability':.5}))
        self.assertNotEqual(canonical({'forecast_at':'a'}),canonical({'forecast_at':'b'}))
    def test_atomic_failure_keeps_previous_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'export.json'
            atomic_bytes(path,b'old')
            with patch('audit_io.os.replace',side_effect=OSError('simulated failure')):
                with self.assertRaises(OSError):
                    atomic_bytes(path,b'new')
            self.assertEqual(path.read_bytes(),b'old')
            self.assertEqual(len(list(Path(folder).iterdir())),1)

    def test_same_week_performance_is_excluded(self):
        per_game=pd.DataFrame({'team':['A','A'],'week':[1,2],'epa':[1.,99.]})
        schedule=pd.DataFrame([dict(game_id='g',season=2026,week=2,home_team='A',away_team='B')])
        result=roll_forward(per_game,schedule,['epa'],'pbp')
        self.assertEqual(result.loc[result.team.eq('A'),'pbp_epa_avg'].iloc[0],1.)
        self.assertTrue(pd.isna(result.loc[result.team.eq('B'),'pbp_epa_avg'].iloc[0]))

    def test_score_distribution_coherence_and_pushes(self):
        residuals=np.tile([2.,-2.],(128,1))
        samples=draws([21,21],residuals)
        self.assertTrue((samples.sum(axis=1)==42).all())
        result=summarize(samples,spread=-4,total=42)
        self.assertEqual(result['home_win']+result['away_win']+result['tie'],1.)
        self.assertEqual(result['spread']['push'],1.)
        self.assertEqual(result['total']['push'],1.)
        self.assertEqual(sum(result['total'].values()),1.)
        np.testing.assert_array_equal(samples,draws([21,21],residuals))
        with self.assertRaises(ValueError):
            draws([21,21],residuals[:10])

    def test_wilson_small_sample(self):
        self.assertEqual(wilson(0,0),[None,None])
        low,high=wilson(1,1)
        self.assertGreater(low,0)
        self.assertLess(low,.3)
        self.assertAlmostEqual(high,1)

    def test_learning_rejects_mixed_versions_and_duplicate_labels(self):
        rows=pd.DataFrame({'game_id':['a','b'],'model_version':['20260101T000000Z','20260201T000000Z']})
        with self.assertRaises(ValueError):
            update_learning(rows,Path('.'))
        rows['game_id']='a'
        with self.assertRaises(ValueError):
            update_learning(rows,Path('.'))

    def test_challenger_minimum_not_bypassed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cohort=root/'learning/generations/20260101T000000Z'
            write_json(cohort/'summary.json',{'candidate_brier':.01,'baseline_brier':.5})
            pd.DataFrame({'game_id':['a','b'],'label':[0,1],'baseline':[.5,.5],'candidate':[.1,.9]}).to_csv(cohort/'evaluation.csv',index=False)
            result=report(root)
            self.assertFalse(result['cohorts'][0]['ready_for_review'])
            self.assertFalse((root/'models').exists())

    def test_staging_failure_leaves_existing_dashboard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'public_site/data/predictions.json'
            write_json(path,[{'game_id':'old','status':'Scheduled'}])
            before=path.read_bytes()
            with patch('refresh_predictions.run',side_effect=RuntimeError('source failure')):
                with self.assertRaises(RuntimeError):
                    refresh(root,2026)
            self.assertEqual(before,path.read_bytes())

    def test_published_data_gate(self):
        root=Path(__file__).resolve().parent
        checked=validate(root)
        self.assertGreater(checked['games'],0)
        data=json.loads((root/'public_site/data/advanced_backtest.json').read_text())
        for fold in data['folds']:
            self.assertLess(fold['inner_train_through'],fold['validation_start'])
            self.assertLess(fold['train_through'],fold['test_cutoff'])

    def test_independent_score_cohorts(self):
        data=json.loads((Path(__file__).resolve().parent/'public_site/data/performance.json').read_text())
        groups=data['score_groups']
        self.assertEqual(len({g['family'] for g in groups}),5)
        for family in {g['family'] for g in groups}:
            selected=[g for g in groups if g['family']==family]
            total=next(g['summary']['score_n'] for g in selected if g['version']=='all')
            self.assertEqual(total,sum(g['summary']['score_n'] for g in selected if g['version']!='all'))


if __name__=='__main__':
    unittest.main()

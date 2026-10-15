import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audit_io import atomic_bytes
from historical_metrics import interval,loss
from research_operations import checkpoints,error_analysis
from release_control import FILES,snapshot,manifest,approve,plan,activate


class Release40Tests(unittest.TestCase):
    def checkpoint_root(self,root,refresh):
        (root/'data').mkdir();(root/'public_site/data').mkdir(parents=True)
        pd.DataFrame([{'game_id':'g','season':2026,'week':6,'game_type':'REG','gameday':'2026-10-18',
                       'gametime':'13:00','home_score':np.nan,'away_score':np.nan}]).to_csv(root/'data/schedule_2026.csv',index=False)
        (root/'public_site/data/predictions.json').write_text(json.dumps([{'game_id':'g','status':'Scheduled','home_win_probability':.6,'model_version':'v'}]))
        (root/'public_site/data/refresh_health.json').write_text(json.dumps({'last_successful_refresh':refresh}))

    def test_checkpoint_exact_window_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.checkpoint_root(root,'2026-10-17T16:55:00Z')
            checkpoints(root,now='2026-10-17T17:00:00Z')
            path=root/'audit/checkpoints.json';before=path.read_bytes()
            saved=json.loads(before);self.assertEqual(list(saved),['g|24h'])
            self.assertEqual(saved['g|24h']['minutes_before_kickoff'],1440)
            checkpoints(root,now='2026-10-17T17:10:00Z')
            self.assertEqual(before,path.read_bytes())
            checkpoints(root,now='2026-10-19T17:00:00Z')
            self.assertEqual(before,path.read_bytes())

    def test_stale_and_missed_not_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.checkpoint_root(root,'2026-10-15T17:00:00Z')
            checkpoints(root,now='2026-10-17T17:00:00Z')
            self.assertEqual(json.loads((root/'audit/checkpoints.json').read_text()),{})
            r=json.loads((root/'public_site/data/checkpoints.json').read_text())
            self.assertEqual(r['windows'][0]['status'],'Needs fresh validated forecast')
            checkpoints(root,now='2026-10-18T17:00:00Z')
            self.assertEqual(json.loads((root/'audit/checkpoints.json').read_text()),{})

    def test_cluster_interval_and_probability_bounds(self):
        self.assertEqual(interval([0.,0.,0.],['w1','w1','w2']),[0.,0.])
        self.assertTrue(np.isfinite(loss(np.array([0.,1.]),np.array([1,0]))).all())

    def test_errors_exclude_ties_and_separate_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'public_site/data').mkdir(parents=True)
            common={'status':'Final','week':1,'home_team':'A','away_team':'B','home_win_probability':.7}
            rows=[dict(common,model_version='a',home_score=20,away_score=10),
                  dict(common,model_version='b',home_score=10,away_score=20),
                  dict(common,model_version='a',home_score=10,away_score=10)]
            (root/'public_site/data/predictions.json').write_text(json.dumps(rows));error_analysis(root)
            d=json.loads((root/'public_site/data/error_analysis.json').read_text())
            confidence=[r for r in d['groups'] if r['dimension']=='confidence']
            self.assertEqual([r['games'] for r in confidence],[1,1])
            self.assertEqual([r['accuracy'] for r in confidence],[1.,0.])

    def bundle(self,root):
        fitted=LogisticRegression().fit(pd.DataFrame({'x':[0.,1.,2.,3.]}),[0,0,1,1])
        for name in FILES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            if name.endswith('.json'):path.write_text('{}')
            else:joblib.dump({'model':fitted,'feature_cols':['x']},path)

    def test_release_requires_approval_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.bundle(root);version=snapshot(root)
            self.assertFalse(plan(root,version)['approved'])
            with self.assertRaises(ValueError):activate(root,version)
            with self.assertRaises(ValueError):manifest(root,'../../escape')
            value=manifest(root,version);sha=value['files'][FILES[0]]
            (root/'.release_blobs'/sha).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):manifest(root,version)

    def test_release_failure_restores_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.bundle(root);baseline=snapshot(root)
            original={name:(root/name).read_bytes() for name in FILES}
            (root/FILES[1]).write_text('{"candidate":true}')
            version=snapshot(root)
            (root/FILES[1]).write_bytes(original[FILES[1]])
            approve(root,version,'Unit test reviewer','Synthetic test only')
            count=0
            def fail_once(path,data):
                nonlocal count
                count+=1
                if count==2:raise OSError('Simulated disk failure')
                return atomic_bytes(path,data)
            with patch('release_control.atomic_bytes',side_effect=fail_once):
                with self.assertRaises(OSError):activate(root,version)
            self.assertTrue(all((root/name).read_bytes()==data for name,data in original.items()))
            self.assertEqual(snapshot(root),baseline)


if __name__=='__main__':unittest.main()

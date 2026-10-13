import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from audit_io import write_json
from dashboard_insights import export, monitor


class InsightTests(unittest.TestCase):
    def game(self):
        return {'game_id':'2026_06_A_B','season':2026,'gameday':'2026-10-18','gametime':'13:00',
                'week':6,'status':'Scheduled','home_score':None,'away_score':None,
                'model_version':'v1','score_model_version':'s1','home_win_probability':.6,
                'predicted_home_score':24,'predicted_away_score':20}

    def test_history_dedup_change_and_final_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            game=self.game()
            path=root/'public_site/data/predictions.json'
            write_json(path,[game])
            at=datetime(2026,10,13,tzinfo=timezone.utc)
            first=export(root,at)
            self.assertEqual(len(first[game['game_id']]),1)
            self.assertEqual(first,export(root,datetime(2026,10,14,tzinfo=timezone.utc)))
            game['home_win_probability']=.7
            write_json(path,[game])
            changed=export(root,datetime(2026,10,15,tzinfo=timezone.utc))
            self.assertEqual(len(changed[game['game_id']]),2)
            game.update(status='Final',home_score=21,away_score=14)
            write_json(path,[game])
            self.assertEqual(changed,export(root,datetime(2026,10,19,tzinfo=timezone.utc)))
            public=json.loads((root/'public_site/data/forecast_history.json').read_text())
            self.assertAlmostEqual(public['games'][game['game_id']][-1]['probability_change'],.1)

    def test_version_horizon_ties_and_late_exclusion(self):
        game=self.game() | {'status':'Final','home_score':20,'away_score':20}
        base={'model_version':'v1','score_model_version':'s1','probability':.6,
              'predicted_home_score':24,'predicted_away_score':20}
        history={game['game_id']:[base|{'forecast_at':'2026-10-13T12:00:00+00:00'},
            base|{'forecast_at':'2026-10-18T16:00:00+00:00','probability':.7},
            base|{'forecast_at':'2026-10-18T18:00:00+00:00','probability':.99},
            base|{'forecast_at':'2026-10-18T15:00:00+00:00','model_version':'v2'}]}
        cohorts=monitor(history,[game])['cohorts']
        latest=[c for c in cohorts if c['horizon']=='Latest pregame']
        self.assertEqual(len(latest),2)
        self.assertTrue(all(c['overall']['n']==0 and c['overall']['ties']==1 for c in latest))
        game.update(home_score=24,away_score=20)
        latest=[c for c in monitor(history,[game])['cohorts'] if c['horizon']=='Latest pregame' and c['model_version']=='v1'][0]
        self.assertAlmostEqual(latest['overall']['brier'],.09)
        self.assertEqual(latest['overall']['score_n'],1)
        self.assertEqual(latest['status'],'Insufficient prospective sample')

    def test_no_retroactive_final_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            game=self.game()|{'status':'Final','home_score':24,'away_score':20}
            write_json(root/'public_site/data/predictions.json',[game])
            self.assertEqual(export(root,datetime(2026,10,19,tzinfo=timezone.utc))[game['game_id']],[])

    def test_monitor_alert_has_minimum_sample(self):
        games, history = [], {}
        for i in range(64):
            game=self.game()|{'game_id':f'game_{i:03}', 'status':'Final','home_score':24,'away_score':20}
            games.append(game)
            history[game['game_id']]=[{'forecast_at':'2026-10-13T12:00:00+00:00',
                'model_version':'v1','score_model_version':None,'probability':.9 if i<32 else .1}]
        latest=next(c for c in monitor(history,games)['cohorts'] if c['horizon']=='Latest pregame')
        self.assertEqual(latest['status'],'Review deterioration')
        self.assertEqual(latest['recent']['n'],32)
        self.assertEqual(latest['baseline']['n'],32)


if __name__=='__main__':
    unittest.main()

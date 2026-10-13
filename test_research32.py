import unittest
import pandas as pd
from research_dataset import lagged
from train_research32 import families, calibrated


class ResearchTests(unittest.TestCase):
    def test_32_configurations(self):
        self.assertEqual(len(families())*2,32)
        self.assertTrue(all(len(v)==2 for v in families().values()))

    def test_current_and_future_stats_excluded(self):
        schedule=pd.DataFrame([{'game_id':str(i),'season':2026,'week':i,'home_team':'A','away_team':'B','home_score':20,'away_score':14} for i in [1,2,3]])
        stats=pd.DataFrame([{'game_id':str(i),'team':t,'rate':i*100} for i in [1,2,3] for t in ['A','B']])
        result=lagged(schedule,stats).set_index('game_id')
        self.assertTrue(pd.isna(result.loc['1','home_research_rate']))
        self.assertEqual(result.loc['2','home_research_rate'],100)
        self.assertEqual(result.loc['3','home_research_rate'],150)

    def test_temporal_overlap_rejected(self):
        from sklearn.linear_model import LogisticRegression
        frame=pd.DataFrame({'gameday':['2026-01-01'],'home_win':[1],'x':[0]})
        with self.assertRaises(ValueError):calibrated(LogisticRegression(),frame,frame,['x'])

import unittest
import numpy as np
import pandas as pd
from matched_backtest import fold_parts, summarize, probability


class MatchedBacktestTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{'game_id':str(year),'season':year,'week':1,'gameday':pd.Timestamp(f'{year}-09-10'),
                              'home_score':21,'away_score':17,'home_win':1} for year in range(2017,2027)])

    def test_temporal_partitions_and_ties(self):
        data=self.fixture()
        tie=data.iloc[-1].copy();tie['game_id']='tie';tie['home_score']=17;tie['home_win']=np.nan
        data=pd.concat([data,pd.DataFrame([tie])],ignore_index=True)
        parts=fold_parts(data,2026)
        self.assertEqual([int(x.season.max()) for x in parts],[2023,2024,2025,2026])
        self.assertEqual(len(parts[-1]),1)
        self.assertTrue(all(a.gameday.max()<b.gameday.min() for a,b in zip(parts,parts[1:])))

    def test_reject_postponed_overlap(self):
        data=self.fixture()
        data.loc[data.season.eq(2025),'gameday']=pd.Timestamp('2026-10-01')
        with self.assertRaises(ValueError): fold_parts(data,2026)

    def test_paired_identity_and_reproducibility(self):
        data=pd.DataFrame({'home_win':[1,0,1,0], 'probability':[.6,.4,.7,.3],
                           'baseline_probability':[.6,.4,.7,.3], 'season':[2025]*4,'week':[1,1,2,2]})
        a=summarize(data,50);b=summarize(data,50)
        self.assertEqual(a,b)
        self.assertEqual(a['delta_log_loss_ci95'],[0.,0.])
        self.assertEqual(a['delta_brier_ci95'],[0.,0.])
        self.assertEqual(a['weeks'],2)

    def test_invalid_probabilities_rejected(self):
        class Invalid:
            def predict_proba(self,x): return np.array([[0.,np.nan]])
        with self.assertRaises(ValueError): probability(Invalid(),pd.DataFrame({'x':[1]}),['x'])

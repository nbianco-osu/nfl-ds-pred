import unittest
import numpy as np
import pandas as pd
from markov_research import fit, simulate, probabilities, mix_paths, choose_weight, extract_drives


def drives():
    return pd.DataFrame([dict(game_id=str(i//10), season=2026, week=1, offense='A', defense='B',
                             field=70., margin=0., remaining=3500., duration=200, own=7 if i%3==0 else 0,
                             opp=0, next_field=70., retain=False) for i in range(160)])


class MarkovTests(unittest.TestCase):
    def test_cutoff_excludes_current_and_future(self):
        past=drives(); future=past.assign(week=2,own=8)
        model=fit(pd.concat([past,future]),2026,2)
        self.assertEqual(len(model['drives']),160)
        with self.assertRaises(ValueError): fit(past,2026,1)

    def test_paths_deterministic_integer_monotone_and_conserved(self):
        model=fit(drives(),2026,2)
        a=simulate(model,'A','B',30)
        np.testing.assert_array_equal(a,simulate(model,'A','B',30))
        self.assertEqual(a.shape,(30,5,2))
        self.assertTrue((np.diff(a,axis=1)>=0).all())
        self.assertTrue(np.issubdtype(a.dtype,np.integer))
        self.assertAlmostEqual(probabilities(a).sum(),1)

    def test_overrun_half_does_not_award_points(self):
        model=fit(drives().assign(duration=2000,own=7),2026,2)
        self.assertEqual(simulate(model,'A','B',5).sum(),0)

    def test_defensive_score_goes_to_opponent(self):
        model=fit(drives().assign(duration=900,own=0,opp=2),2026,2)
        result=simulate(model,'A','B',10)
        np.testing.assert_array_equal(result[:,-1],np.full((10,2),4))

    def test_mixture_preserves_whole_paths(self):
        a=np.zeros((30,5,2),int);b=np.ones_like(a)*7
        result=mix_paths(a,b,.5)
        self.assertTrue(all(np.array_equal(x,a[0]) or np.array_equal(x,b[0]) for x in result))
        np.testing.assert_array_equal(mix_paths(a,b,0),a)
        np.testing.assert_array_equal(mix_paths(a,b,1),b)

    def test_weight_uses_supplied_earlier_oof_only(self):
        self.assertEqual(choose_weight([])[0],.5)
        prior=[dict(team=[1,0,0],league=[0,0,1],outcome=0)]*16
        self.assertEqual(choose_weight(prior)[0],1)

    def test_invalid_trials_and_schema_fail_closed(self):
        with self.assertRaises(ValueError):simulate(fit(drives(),2026,2),'A','B',0)
        with self.assertRaises(ValueError):extract_drives(pd.DataFrame())

    def test_drive_extraction_retains_joint_clock_and_score(self):
        common=dict(game_id='g',season=2026,week=1,qtr=1,fixed_drive=1,posteam='A',defteam='B',
                    yardline_100=70,posteam_score=0,defteam_score=0,defteam_score_post=0)
        p=pd.DataFrame([{**common,'play_id':1,'game_seconds_remaining':3500,'posteam_score_post':0},
                        {**common,'play_id':2,'game_seconds_remaining':3300,'posteam_score_post':7}])
        row=extract_drives(p).iloc[0]
        self.assertEqual(row.duration,200);self.assertEqual(row.own,7);self.assertEqual(row.opp,0)


if __name__=='__main__':unittest.main()

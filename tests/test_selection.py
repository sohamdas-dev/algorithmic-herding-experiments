import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policy_selection import choose
from calibrate import batches

class SelectionTests(unittest.TestCase):
    def test_infeasible_fallback_is_not_called_feasible(self):
        def row(h,d,q=.01):return dict(herding=h,delta=d,quality_worst=q,deadline_completion=.99)
        s={0:{'a':row(1.6,0),'b':row(1.4,.2),'full':row(.1,0)}}
        target=dict(herding_mean=1.25,quality_worst_mean=.04,deadline_completion_mean=.98)
        got=choose(s,[0],target)
        self.assertEqual(got['policy'],'b');self.assertFalse(got['training_target_met'])
        s[0]['a']=row(1.2,0);s[0]['b']=row(1.1,.2)
        self.assertEqual(choose(s,[0],target)['policy'],'a')
        s[0]['a']['quality_worst']=.2
        self.assertEqual(choose(s,[0],target)['policy'],'b')
    def test_batch_construction_filters_and_orders(self):
        features=[dict(seed=i,memory=m,policy=p,feature=[i,2*i]) for i in [4,1,3,2] for m in [0,4] for p in ['a','b']]
        got=batches(features,4,'b',2)
        np.testing.assert_equal(got,[[1.5,3],[3.5,7]])
if __name__=='__main__':unittest.main()

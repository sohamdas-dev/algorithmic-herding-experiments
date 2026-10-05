import sys,unittest
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model import *

class ModelTests(unittest.TestCase):
    def test_transition_probabilities_and_expected_pressure(self):
        c=Config(); P=transition(c)
        self.assertTrue(np.all(P>=0)); np.testing.assert_allclose(P.sum(-1),1)
        for x,a,k in [(0,0,0),(2,1,7),(4,1,23)]:
            wanted=(1-c.physical_noise)*(c.persistence*x+(1-c.persistence)*(c.state_levels-1)*(k+a)/c.n)+c.physical_noise*(c.state_levels-1)/2
            self.assertAlmostEqual(P[x,a,k]@np.arange(c.state_levels),wanted)
    def test_bellman_against_recursive_enumeration(self):
        c=replace(Config(),n=2,groups=2,state_levels=3,horizon=3,deadline=2,demand=2)
        p=prior(c);P=transition(c); hold=np.ones(2)
        def value(t,r,x):
            if t==c.horizon: return c.terminal_cost*r
            return min(q(t,r,x,a) for a in range(min(r,1)+1))
        def q(t,r,x,a):
            out=(c.holding_cost+(c.overdue_cost if t>=c.deadline else 0))*r
            for k in range(c.n):
                out+=p[0,x,k]*(a*(c.energy_cost+c.congestion*(k+a)/c.n+c.pressure_cost*x/(c.state_levels-1))+c.discount*sum(P[x,a,k,y]*value(t+1,r-a,y) for y in range(c.state_levels)))
            return out
        got=action_values(c,p,P,0,np.array([2,2]),1,hold)
        np.testing.assert_allclose(got[0],[q(0,2,1,0),q(0,2,1,1)],atol=1e-12)
    def test_service_conservation_reproducibility_and_action_set(self):
        c=Config();pol=Policy('staggered',(0,1,3,4)); a=simulate(c,pol,54);b=simulate(c,pol,54)
        for key in a: np.testing.assert_equal(a[key],b[key])
        self.assertTrue(np.isin(a['action'],[0,1]).all())
        np.testing.assert_array_equal(a['backlog'][0]-a['backlog'][-1],a['action'].sum(0))
        self.assertTrue((a['backlog']>=0).all())
        self.assertTrue(np.isfinite(a['quality_loss']).all())
        self.assertTrue((a['quality_loss']>=-1e-12).all())
    def test_record_count_and_window(self):
        c=Config();p=prior(c); records=np.zeros((4,c.n,c.n)); records[:,:,2]=1
        states=np.array([1,1,1,1]);got=make_beliefs(c,records,states,4,2,p)
        expected=(c.prior_strength*p[:,1]+records[:2].sum(0))/(c.prior_strength+2)
        np.testing.assert_allclose(got[:,1],expected)
        records[0,:,2]=0;records[0,:,10]=1
        np.testing.assert_allclose(make_beliefs(c,records,states,4,2,p),got)
    def test_equal_delivery_counts_and_public_inference(self):
        c=Config(); t=simulate(c,Policy('staggered',(0,1,3,4)),4,quality=False)
        np.testing.assert_equal(t['delivery'][1:c.deadline+1].sum(0),np.full(c.n,c.deadline//c.period))
        pub=simulate(c,Policy('public_only',mode='public_only'),4,quality=False)
        fro=simulate(c,Policy('frozen',mode='frozen'),4,quality=False)
        self.assertFalse(np.allclose(pub['belief_mean'],fro['belief_mean']))
        self.assertEqual(metrics(c,fro,fro)['herding'],0)

if __name__=='__main__': unittest.main()

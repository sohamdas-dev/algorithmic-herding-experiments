"""Finite-action stochastic aggregative game and observation-aware empirical learner.

All arrays and randomness have explicit semantics; see MODEL.md. No potential-game
or convergence claim is made. Dynamic programming is exact for the current frozen
state-conditioned opponent model, not for the jointly adapting population.
"""
from dataclasses import dataclass, asdict
import numpy as np
from math import comb

@dataclass(frozen=True)
class Config:
    n: int = 24
    groups: int = 4
    horizon: int = 42
    deadline: int = 30
    demand: int = 8
    state_levels: int = 5
    period: int = 6
    persistence: float = .65
    physical_noise: float = .08
    congestion: float = 2.0
    pressure_cost: float = 1.0
    energy_cost: float = .15
    holding_cost: float = .025
    overdue_cost: float = .5
    terminal_cost: float = 8.0
    discount: float = .98
    prior_mean: float = .30
    prior_strength: float = 3.0
    temperature: float = .025
    heterogeneity: float = .03
    initial_state: int = 1

@dataclass(frozen=True)
class Policy:
    name: str
    phases: tuple = (0,0,0,0)
    mode: str = 'periodic'  # periodic, randomized, full, public_only, frozen


def transition(c):
    """P[x,a,k,xnext], k is the INTEGER number of active opponents."""
    X=c.state_levels; n=c.n
    P=np.zeros((X,2,n,X))
    for x in range(X):
        for a in (0,1):
            for k in range(n):
                z=c.persistence*x+(1-c.persistence)*(X-1)*(k+a)/n
                lo=int(np.floor(z)); hi=min(lo+1,X-1)
                P[x,a,k,lo]+=1-(z-lo); P[x,a,k,hi]+=z-lo
    # Uniform exogenous mixing prevents an exact inversion of the physical state.
    return (1-c.physical_noise)*P+c.physical_noise/X


def prior(c):
    k=np.arange(c.n)
    p=np.array([comb(c.n-1,int(j))*c.prior_mean**j*(1-c.prior_mean)**(c.n-1-j) for j in k])
    return np.broadcast_to(p,(c.n,c.state_levels,c.n)).copy()


def action_values(c,belief,P,t,remaining,x,hold_factor):
    """Exact finite-horizon Bellman recursion under a frozen opponent belief.
    Remaining service is fully observed. Other-agent activity at each future state
    is drawn from the empirical conditional distribution currently held by agent i.
    """
    n,X=c.n,c.state_levels; R=c.demand+1
    trans=np.einsum('ixk,xaky->iaxy',belief,P,optimize=False)
    mean=np.einsum('ixk,k->ix',belief,np.arange(n))
    active=c.energy_cost+c.congestion*(mean+1)/n+c.pressure_cost*np.arange(X)[None,:]/(X-1)
    r=np.arange(R)[None,:,None]
    V=np.broadcast_to(c.terminal_cost*r,(n,R,X)).copy()
    for future in range(c.horizon-1,t-1,-1):
        hold=(c.holding_cost+(c.overdue_cost if future>=c.deadline else 0))*hold_factor[:,None,None]*r
        q0=hold+c.discount*np.einsum('ixy,iry->irx',trans[:,0],V,optimize=False)
        served=V[:,np.maximum(np.arange(R)-1,0),:]
        q1=hold+active[:,None,:]+c.discount*np.einsum('ixy,iry->irx',trans[:,1],served,optimize=False)
        q1[:,0,:]=np.inf
        V=np.minimum(q0,q1)
    idx=np.arange(n)
    return np.stack([q0[idx,remaining,x],q1[idx,remaining,x]],axis=1)


def make_beliefs(c,records,state_history,t,memory,p0):
    if t==0: return p0.copy()
    ages=t-1-np.arange(t)
    weights=np.ones(t) if memory<=0 else (ages<memory).astype(float)
    # Prior is retained, every time record has total mass exactly one.
    counts=c.prior_strength*p0.copy()
    for x in range(c.state_levels):
        w=weights*(state_history[:t]==x)
        counts[:,x,:]+=np.einsum('s,sik->ik',w,records[:t],optimize=False)
    return counts/counts.sum(axis=2,keepdims=True)


def simulate(c,policy,seed,memory=0,quality=True,physical_override=None,ignore_public=False):
    rng=np.random.default_rng(seed)
    draws=rng.random((c.horizon,c.n))
    physics=rng.random(c.horizon)
    hold_factor=np.exp(c.heterogeneity*rng.standard_normal(c.n))
    group=np.repeat(np.arange(c.groups),c.n//c.groups)
    # Private delivery randomness is independent of environmental / action randomness.
    prng=np.random.default_rng(seed+891731)
    phases=prng.integers(0,c.period,c.n) if policy.mode=='randomized' else np.array(policy.phases)[group]
    P=transition(c); Pactual=P if physical_override is None else transition(physical_override)
    H,n,X=c.horizon,c.n,c.state_levels
    p0=prior(c); records=np.zeros((H,n,n)); true_records=np.zeros_like(records)
    states=np.zeros(H+1,dtype=int); states[0]=c.initial_state
    remaining=np.full(n,c.demand,dtype=int)
    actions=np.zeros((H,n),dtype=np.int8); margins=np.zeros((H,n)); means=np.zeros((H,n))
    delivery=np.zeros((H,n),dtype=bool); loss=np.zeros((H,n)); costs=np.zeros((H,n))
    backlog=np.zeros((H+1,n),dtype=np.int16); backlog[0]=remaining
    gaps=np.zeros((H,n)); infos=np.zeros((H,n))
    for t in range(H):
        if t and not ignore_public:
            # Every agent sees the public physical transition and its own action.
            # Bayesian soft counts prevent the experiment from ignoring this information.
            likelihood=P[states[t-1],actions[t-1],:,states[t]]
            post=previous_belief[np.arange(n),states[t-1],:]*likelihood
            post/=post.sum(axis=1,keepdims=True)
            records[t-1]=post
        receive=((t-1)%c.period==phases)&(t>0)
        if policy.mode=='full' or t>c.deadline: receive=np.full(n,t>0)
        if policy.mode in ('public_only','frozen'): receive[:]=False
        delivery[t]=receive
        if t and np.any(receive): records[:t,receive,:]=true_records[:t,receive,:]
        belief=make_beliefs(c,records,states,t,memory,p0)
        if policy.mode=='frozen': belief=p0.copy()
        previous_belief=belief
        x=states[t]
        means[t]=belief[:,x,:]@np.arange(n)/(n-1)
        q=action_values(c,belief,P,t,remaining,x,hold_factor)
        margin=q[:,0]-q[:,1]
        margins[t]=np.where(remaining>0,margin,np.nan)
        # Independent entropy-smoothed choices. Temperature has cost units.
        prob=1/(1+np.exp(-np.clip(margin/c.temperature,-40,40)))
        prob[remaining==0]=0
        actions[t]=(draws[t]<prob).astype(np.int8)
        gaps[t]=np.where(remaining>0,np.abs(margin),0)
        if quality:
            full=make_beliefs(c,true_records,states,t,memory,p0)
            fq=action_values(c,full,P,t,remaining,x,hold_factor)
            # Expected one-decision reference loss; avoids a lucky action realization.
            loss[t]=(1-prob)*(fq[:,0]-fq.min(axis=1))+prob*np.where(np.isfinite(fq[:,1]),fq[:,1]-fq.min(axis=1),0)
        a=actions[t]; total=int(a.sum()); k=total-a
        true_records[t,np.arange(n),k]=1
        costs[t]=(c.holding_cost+(c.overdue_cost if t>=c.deadline else 0))*hold_factor*remaining+a*(c.energy_cost+c.congestion*total/n+c.pressure_cost*x/(X-1))
        remaining=remaining-a; backlog[t+1]=remaining
        px=Pactual[x,0,total,:] if total<n else Pactual[x,1,n-1,:]
        states[t+1]=min(int(np.searchsorted(np.cumsum(px),physics[t],side='right')),X-1)
    costs[-1]+=c.terminal_cost*remaining
    load=actions.mean(axis=1)
    change=np.diff(actions.astype(float),axis=0,prepend=np.zeros((1,n)))
    mass=np.abs(change).sum(axis=1)
    alignment=np.divide(np.abs(change.sum(axis=1)),mass,out=np.zeros(H),where=mass>0)
    result=dict(load=load,state=states,action=actions,margin=margins,belief_mean=means,delivery=delivery,
                quality_loss=loss,cost=costs,backlog=backlog,alignment=alignment,participation=mass/n,group=group)
    return result


def metrics(c,tr,reference=None):
    load=tr['load']; steps=np.r_[0,load]; L=6
    raw=max(float(np.max(steps[l:]-steps[:-l])) for l in range(1,L+1))
    herd=np.nan
    if reference is not None:
        ref=np.r_[0,reference['load']]
        herd=max(0.,max(float(np.max((steps[l:]-steps[:-l])-(ref[l:]-ref[:-l]))) for l in range(1,L+1)))
    return dict(herding=herd,peak=float(load.max()),buildup=raw,
                mean_cost=float(tr['cost'].sum(axis=0).mean()),
                quality_mean=float(tr['quality_loss'].mean()),quality_worst=float(tr['quality_loss'].mean(axis=0).max()),
                completion=float(1-tr['backlog'][-1].sum()/(c.n*c.demand)),
                deadline_completion=float(1-tr['backlog'][c.deadline].sum()/(c.n*c.demand)),
                tail_peak=float(load[c.deadline:].max()),
                switching=float(tr['participation'].sum()),
                aligned_mass=float(np.mean(tr['alignment']*tr['participation'])))

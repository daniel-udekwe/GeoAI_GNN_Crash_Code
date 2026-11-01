import numpy as np, json, time, sys, statsmodels.api as sm, lightgbm as lgb
import gm, ev
from gm import TR,VA,TE
tag=sys.argv[1] if len(sys.argv)>1 else '0p5'; which=sys.argv[2] if len(sys.argv)>2 else 'all'
SUB=sys.argv[3].split(",") if len(sys.argv)>3 else None
D=gm.Data(tag); E=ev.Evaluator(D); R={}; preds={}
def save():
    json.dump(R,open(f'res_main_{tag}.json','w'),indent=1)
    np.savez_compressed(f'preds_main_{tag}.npz',**{k:np.stack([v[t] for t in TE],1).astype(np.float32) for k,v in preds.items()})
def rec(name,P,extra=None):
    r=E.all(P); r.update(extra or {}); R[name]=r; preds[name]=P
    print(name,{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items() if k in ['dev','skill','pai1','moran','skill_rural','skill_urban','calib']},flush=True); save()
rec('HA',E.HA)
# NB GLMs
def Xmat(t,lags=True,spatial=False):
    X=D.X[t]; drop={5,16}; cols=[c for c in (range(5,X.shape[1]) if not lags else range(X.shape[1])) if c not in drop]
    X=X[:,cols]
    if spatial: X=np.c_[X,D.Xs[t]]
    return X
rng=np.random.default_rng(0)
for nm,lags in [x for x in [('NB-SPF',False),('NB-lag',True)] if SUB is None or x[0] in SUB]:
    Xt=np.concatenate([Xmat(t,lags) for t in TR]); yt=np.concatenate([D.O[:,t] for t in TR])
    idx=rng.choice(len(yt),500000,replace=False); Xs=sm.add_constant(Xt[idx],has_constant='add'); ys=yt[idx]
    keep=Xs.std(0)>0; keep[0]=True; Xs=Xs[:,keep]
    p=sm.GLM(ys,Xs,family=sm.families.Poisson()).fit(); mu=p.mu; a=float(np.sum((((ys-mu)**2-ys)/mu)*mu)/np.sum(mu*mu))
    nb=sm.GLM(ys,Xs,family=sm.families.NegativeBinomial(alpha=a)).fit()
    P={t:np.exp(sm.add_constant(Xmat(t,lags),has_constant='add')[:,keep]@nb.params).astype(np.float32) for t in range(4,16)}
    rec(nm,P,{'alpha':a})
    if nm=='NB-SPF': np.savez_compressed(f'spf_{tag}.npz',**{str(t):P[t] for t in P})
# LightGBM
for nm,sp_ in [x for x in [('LGBM',False),('LGBM-S',True)] if SUB is None or x[0] in SUB]:
    Xm=lambda t,sp_=sp_: np.c_[D.X[t],D.Xs[t]] if sp_ else D.X[t]
    Xt=np.concatenate([Xm(t) for t in TR]); yt=np.concatenate([D.O[:,t] for t in TR])
    Xv=np.concatenate([Xm(t) for t in VA]); yv=np.concatenate([D.O[:,t] for t in VA])
    m=lgb.LGBMRegressor(objective='poisson',n_estimators=2000,learning_rate=0.05,num_leaves=63,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
    m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(50,verbose=False)])
    P={t:m.predict(Xm(t)).astype(np.float32) for t in range(4,16)}
    imp=m.booster_.feature_importance('gain'); rec(nm,P,{'iters':int(m.best_iteration_),'gain_spatial_share':float(imp[-6:].sum()/imp.sum()) if sp_ else 0.0})
# neural models
cfg=[('MLP','mlp',False,False),('GRU','gru',False,False),('GCN','gcn',False,False),('GAT','gat',False,False),('ST-GNN','stgnn',False,False),('EI-STGNN','stgnn',True,True)]
seeds=[0,1,2] if SUB is None else [0]
for nm,kind,nb,off in [c for c in cfg if SUB is None or c[0] in SUB]:
    Ps=[]; eps=[]; t0=time.time()
    for sd in (seeds if kind!='gat' else [0,1]):
        m,P,ep=gm.train(D,kind,seed=sd,epochs=60,lr=1e-2,nb=nb,offset=off,patience=8); Ps.append(P); eps.append(ep)
        R.setdefault(nm+'_seeds',[]).append(E.all(P)['dev'])
        if kind=='gat' and sd==0:
            import torch
            a=m.layers[0].last.mean(1).numpy(); np.save(f'gat_att_{tag}.npy',a)
        if nm=='EI-STGNN' and sd==0:
            import torch; torch.save(m.state_dict(),f'ei_{tag}.pt')
    P={t:np.mean([p[t] for p in Ps],0) for t in range(4,16)}
    rec(nm,P,{'epochs':eps,'secs':round(time.time()-t0),'alpha':float(np.exp(m.logalpha.item())) if nb else None})
print('done')

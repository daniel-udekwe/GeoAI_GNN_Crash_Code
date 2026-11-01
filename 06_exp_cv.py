import numpy as np, json, time, lightgbm as lgb
import gm, ev
from gm import TR,VA,TE
D=gm.Data('0p5'); Ev=ev.Evaluator(D); R={}
rng=np.random.default_rng(7); N=D.N; K=4
rand_fold=rng.integers(0,K,N)
cty=np.unique(D.county[~np.isnan(D.county)]); perm=rng.permutation(cty); cmap={c:i%K for i,c in enumerate(perm)}
blk_fold=np.array([cmap.get(c,0) for c in D.county])
def devm(P,m): return np.mean([gm.dev(P[t].astype(np.float64),D.O[:,t].astype(np.float64))[m].mean() for t in TE])
for scheme,fold in [('random',rand_fold),('county-blocked',blk_fold)]:
    for model in ['LGBM','LGBM-S','MLP','GCN']:
        sk=[]; t0=time.time()
        for f in range(K):
            tr=fold!=f; te=fold==f
            if model.startswith('LGBM'):
                Xm=(lambda t: np.c_[D.X[t],D.Xs[t]]) if model=='LGBM-S' else (lambda t: D.X[t])
                Xt=np.concatenate([Xm(t)[tr] for t in TR]); yt=np.concatenate([D.O[tr,t] for t in TR])
                Xv=np.concatenate([Xm(t)[tr] for t in VA]); yv=np.concatenate([D.O[tr,t] for t in VA])
                m=lgb.LGBMRegressor(objective='poisson',n_estimators=2000,learning_rate=0.05,num_leaves=63,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
                m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(50,verbose=False)])
                P={t:np.maximum(m.predict(Xm(t)),1e-4) for t in TE}
            else:
                kind={'MLP':'mlp','GCN':'gcn','EI-STGNN':'stgnn'}[model]; ei=model=='EI-STGNN'
                _,P,_=gm.train(D,kind,seed=0,epochs=60,lr=1e-2,nb=ei,offset=ei,mask=tr,patience=8)
            sk.append(1-devm(P,te)/devm(Ev.HA,te))
        R[f'{scheme}|{model}']={'skill_mean':float(np.mean(sk)),'skill_sd':float(np.std(sk)),'folds':[float(x) for x in sk],'secs':round(time.time()-t0)}
        print(scheme,model,round(np.mean(sk),4),round(np.std(sk),4),flush=True)
        json.dump(R,open('res_cv.json','w'),indent=1)
print('done')

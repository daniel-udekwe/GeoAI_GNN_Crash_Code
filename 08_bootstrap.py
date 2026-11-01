import numpy as np, json
d=np.load('data_0p5.npz'); P=np.load('preds_main_0p5.npz'); O=d['O'][:,12:16].astype(float); cty=d['county']
def dev(mu,y): return 2*(np.where(y>0,y*np.log(np.where(y>0,y,1)/mu),0)-(y-mu))
D={k:dev(np.maximum(P[k].astype(float),1e-6),O).sum(1) for k in P.files}  # per-node test deviance sum
cs=np.unique(cty[~np.isnan(cty)]); idx={c:np.where(cty==c)[0] for c in cs}
rng=np.random.default_rng(0); B=2000
pairs=[('GCN','MLP'),('GAT','MLP'),('ST-GNN','GRU'),('LGBM-S','LGBM'),('EI-STGNN','ST-GNN'),('ST-GNN','LGBM-S'),('EI-STGNN','LGBM-S'),('GCN','LGBM-S')]
csum={k:np.array([D[k][idx[c]].sum() for c in cs]) for k in D}
out={}
for a,b in pairs:
    da=csum[a]; db=csum[b]; h=csum['HA']
    point=(db.sum()-da.sum())/h.sum()
    bs=[]
    for _ in range(B):
        s=rng.integers(0,len(cs),len(cs)); bs.append((db[s].sum()-da[s].sum())/h[s].sum())
    lo,hi=np.percentile(bs,[2.5,97.5]); out[f'{a} vs {b}']=[float(point),float(lo),float(hi)]
    print(a,b,round(point,4),round(lo,4),round(hi,4))
json.dump(out,open('boot.json','w'),indent=1)

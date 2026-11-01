import numpy as np, json, pandas as pd
G=''  # run from the working directory that holds the experiment outputs
d=np.load(G+'data_0p5.npz',allow_pickle=True); P=np.load(G+'preds_main_0p5.npz')
O=d['O']; N=O.shape[0]; cty=d['county']; L=d['length']; obs=d['obs'][:,3]>0.5
Y=O[:,12:16].astype(float)
def dev(mu,y): return 2*(np.where(y>0,y*np.log(np.where(y>0,y,1)/mu),0)-(y-mu))
D={k:dev(np.maximum(P[k].astype(float),1e-6),Y).sum(1) for k in P.files}
df=pd.DataFrame({'cty':cty,'L':L,'obs':obs,'crash4':O.sum(1),'urban':d['urban']})
for k in D: df['dev_'+k]=D[k]
g=df.groupby('cty').agg(miles=('L','sum'),imp=('obs',lambda s:1-s.mean()),crash=('crash4','sum'),urban=('urban','mean'),**{f'dev_{k}':(f'dev_{k}','sum') for k in D})
for k in D: g['skill_'+k]=1-g['dev_'+k]/g['dev_HA']
g['density']=g.crash/g.miles
# CV folds reproduce
rng=np.random.default_rng(7); _=rng.integers(0,4,N)
cs=np.unique(cty[~np.isnan(cty)]); perm=rng.permutation(cs); cmap={c:i%4 for i,c in enumerate(perm)}
g['fold']=[cmap[c] for c in g.index]
R=json.load(open(G+'res_cv.json')); g['fold_skill_lgbms']=[R['county-blocked|LGBM-S']['folds'][f] for f in g.fold]
g['gain_gcn']=g.skill_GCN-g.skill_MLP; g['gain_lgbms']=g['skill_LGBM-S']-g.skill_LGBM
g['fips']=[f'37{int(2*c-1):03d}' for c in g.index]
g.to_csv('county_metrics.csv')
print(g[['miles','crash','density','imp','skill_LGBM-S','gain_gcn','fold','fold_skill_lgbms']].describe().round(3))
print('corr gain_gcn vs imp',np.corrcoef(g.gain_gcn,g.imp)[0,1].round(3),'vs density',np.corrcoef(g.gain_gcn,np.log(g.density))[0,1].round(3), 'skill vs urban', np.corrcoef(g['skill_LGBM-S'],g.urban)[0,1].round(3))
print('counties gcn>mlp',(g.gain_gcn>0).sum())
# class x model heatmap
classes=list(d['classes']); st=d['static']; cls=st[:,4:12].argmax(1)
rows=[]
for ci,c in enumerate(classes):
    m=cls==ci; r={'class':c,'n':int(m.sum())}
    for k in D:
        if k!='HA': r[k]=1-D[k][m].sum()/D['HA'][m].sum()
    rows.append(r)
H=pd.DataFrame(rows); H.to_csv('class_skill.csv',index=False); print(H.round(3).to_string())

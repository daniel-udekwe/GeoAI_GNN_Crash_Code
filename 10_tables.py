import json, numpy as np, os
import os; os.makedirs('parts',exist_ok=True)
G=''  # run from the working directory that holds the experiment outputs
M=json.load(open(G+'res_main_0p5.json')); dh=M['HA']['dev']
order=['HA','NB-SPF','NB-lag','LGBM','LGBM-S','MLP','GRU','GCN','GAT','ST-GNN','EI-STGNN']
def sk_sd(m):
    s=M.get(m+'_seeds'); 
    if not s or len(s)<2: return ''
    ss=[1-d/dh for d in s]; return f' ({min(ss):.3f}--{max(ss):.3f})'
best=lambda k,f=max: f(M[m][k] for m in order if m!='HA')
rows=[]
for m in order:
    r=M[m]
    rows.append(f"{m} & {r['dev']:.4f} & {r['skill']:.3f}{sk_sd(m)} & {r['mae']:.4f} & {r['rmse']:.3f} & {r['calib']:.3f} & {r['spearman']:.3f} & {r['cap1']*100:.1f} & {r['cap5']*100:.1f} & {r['ka1']*100:.1f} & {r['moran']:.3f} \\\\")
open('parts/tab_main.tex','w').write('\n'.join(rows))
rows=[]
for m in order[1:]:
    r=M[m]
    rows.append(f"{m} & {r['skill_urban']:.3f} & {r['skill_rural']:.3f} & {r['skill_rural']-r['skill_urban']:+.3f} & {r['skill_aadt_obs']:.3f} & {r['skill_aadt_imp']:.3f} & {r['skill_aadt_imp']-r['skill_aadt_obs']:+.3f} & {r['calib_rural']:.3f} & {r['calib_aadt_imp']:.3f} & {r['rural_sel1']*100:.1f} \\\\")
open('parts/tab_groups.tex','w').write('\n'.join(rows))
print(open('parts/tab_main.tex').read()); print(open('parts/tab_groups.tex').read())
for a,b in [('GCN','MLP'),('GAT','MLP'),('ST-GNN','GRU'),('LGBM-S','LGBM'),('GRU','MLP'),('ST-GNN','GCN'),('EI-STGNN','ST-GNN')]:
    print(a,'-',b, round(M[a]['skill']-M[b]['skill'],4), 'moran',round(M[a]['moran'],3),round(M[b]['moran'],3))
for m in ['MLP','GRU','GCN','GAT','ST-GNN','EI-STGNN']:
    s=[1-d/dh for d in M[m+'_seeds']]; print(m,'seed sd',round(np.std(s),4),'range',round(max(s)-min(s),4))

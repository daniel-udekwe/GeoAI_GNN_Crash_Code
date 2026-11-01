import numpy as np, torch, json, scipy.sparse as sp
from scipy.stats import spearmanr
import gm
D=gm.Data('0p5'); out={}
# hop distance rings from sampled nodes via BFS on adjacency
A=D.A.tocsr()
def rings(i,maxh=4):
    seen={i:0}; front=[i]
    for h in range(1,maxh+1):
        nxt=[]
        for u in front:
            for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
                if v not in seen: seen[v]=h; nxt.append(v)
        front=nxt
    return seen
rng=np.random.default_rng(3)
cand=np.where(D.dg>=2)[0]; S=rng.choice(cand,300,replace=False)
for name,K,f in [('K=2','2','ei_0p5.pt'),('K=4','4','ei_K4.pt')]:
    try: sd=torch.load(f)
    except Exception as e: print('missing',f); continue
    m=gm.Net('stgnn',D.F,64,int(K),True,True); m.load_state_dict(sd); m.eval()
    t=12; x=D.x(t).requires_grad_(True); sq=D.seq(t).requires_grad_(True); off=torch.tensor(D.off[t])
    prof=np.zeros(6); cnt=0
    for i in S:
        if x.grad is not None: x.grad=None; sq.grad=None
        eta=m(x,sq,off,D); eta[i].backward()
        g=x.grad.abs().sum(1)+sq.grad.abs().sum((1,2)); g=g.numpy(); tot=g.sum()
        if tot<=0: continue
        rg=rings(i); p=np.zeros(6)
        for j,h in rg.items(): p[h]+=g[j]
        p[5]=tot-p[:5].sum()
        prof+=p/tot; cnt+=1
    out[f'influence_{name}']=(prof/cnt).tolist(); print(name,np.round(prof/cnt,4),flush=True)
# attention
try:
    a=np.load('gat_att_0p5.npy'); E=D.E; et=D.d['et']; N=D.N
    src=np.r_[E[:,0],E[:,1],np.arange(N)]; dst=np.r_[E[:,1],E[:,0],np.arange(N)]; typ=np.r_[et,et,np.full(N,2)]
    deg=np.bincount(dst,minlength=N)  # incl self
    rel=a*deg[dst]  # attention relative to uniform
    res={}
    for k,nm in [(0,'route'),(1,'junction'),(2,'self')]:
        m=typ==k; res[nm]={'mean_att':float(a[m].mean()),'mean_rel_uniform':float(rel[m].mean()),'n':int(m.sum())}
    la=np.log(np.clip(D.d['aadt'][:,3],10,None)); diff=la[src]-la[dst]; nm=typ<2
    res['spearman_rel_vs_logAADT_ratio']=float(spearmanr(rel[nm],diff[nm]).statistic)
    # by quintile of traffic contrast
    from scipy.stats import rankdata
    rk=rankdata(diff[nm])/nm.sum(); b=np.minimum((rk*5).astype(int),4)
    res['rel_by_contrast_quintile']=[float(rel[nm][b==k].mean()) for k in range(5)]
    res['contrast_quintile_bounds']=[float(np.quantile(diff[nm],p)) for p in [0.2,0.4,0.6,0.8]]
    hasn=deg[dst]>1
    for k,nm2 in [(0,'route'),(1,'junction'),(2,'self')]:
        m=(typ==k)&hasn; res[nm2+'_nonisolated']={'mean_rel_uniform':float(rel[m].mean()),'p10':float(np.quantile(rel[m],0.1)),'p90':float(np.quantile(rel[m],0.9))}
    res['share_rel_gt_1p5']=float((rel[hasn]>1.5).mean()); res['share_rel_lt_0p5']=float((rel[hasn]<0.5).mean())
    O=D.O[:,:12].sum(1); lc=np.log1p(O); dc=lc[src]-lc[dst]
    res['spearman_rel_vs_logcount_diff']=float(spearmanr(rel[nm],dc[nm]).statistic)
    # nodes with deg>=2 neighbours: entropy relative to uniform
    out['attention']=res; print(json.dumps(res,indent=1))
except Exception as e: print('att err',e)
json.dump(out,open('diag.json','w'),indent=1)

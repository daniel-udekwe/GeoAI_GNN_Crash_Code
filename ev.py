import numpy as np, scipy.sparse as sp
from gm import TE, TR, dev
class Evaluator:
    def __init__(s,D):
        s.D=D; O=D.O; N=D.N
        # HA baseline with seasonal factor from training targets
        ratio=np.zeros(4); cnt=np.zeros(4)
        for t in TR: ratio[t%4]+=O[:,t].sum()/O[:,t-4:t].mean(1).sum(); cnt[t%4]+=1
        s.sf=ratio/np.maximum(cnt,1)
        for q in range(4):
            if cnt[q]==0: s.sf[q]=1
        s.HA={t:np.maximum(O[:,t-4:t].mean(1)*s.sf[t%4],1e-3) for t in range(4,16)}
        s.Y=np.stack([O[:,t] for t in TE],1); s.Ya=s.Y.sum(1)
        s.KAa=np.stack([D.KA[:,t] for t in TE],1).sum(1)
        A=D.A.copy(); A.setdiag(0); A.eliminate_zeros(); s.A=A
        s.has=D.dg>0
        # hop rings up to 4
        R=[]; reach=sp.eye(N,format='csr',dtype=bool); cur=sp.eye(N,format='csr',dtype=bool); Ab=A.astype(bool)
        for k in range(1,5):
            cur=(cur@Ab).astype(bool); ring=cur.astype(np.int8)-cur.multiply(reach).astype(np.int8); ring.eliminate_zeros(); ring=ring.astype(bool)
            R.append(ring.tocsr()); reach=(reach+cur).astype(bool)
        s.R=R
        s.groups={'urban':D.urban,'rural':~D.urban,'aadt_obs':D.obs[:,3]>0.5,'aadt_imp':D.obs[:,3]<=0.5}
        s.devHA=s._dev(s.HA)
    def _dev(s,P,mask=None):
        d=np.stack([dev(P[t].astype(np.float64),s.D.O[:,t].astype(np.float64)) for t in TE],1)
        return d.mean() if mask is None else d[mask].mean()
    def resid(s,P):
        Yh=np.stack([P[t] for t in TE],1).sum(1); return (s.Ya-Yh)/np.sqrt(np.maximum(Yh,1e-3)), Yh
    def moran(s,r):
        m=s.has; z=r-r[m].mean(); z[~m]=0
        Wr=sp.diags(1/np.maximum(np.asarray(s.A.sum(1)).ravel(),1))@s.A
        return float((z[m]*(Wr@z)[m]).sum()/(z[m]**2).sum())
    def correlogram(s,r):
        out=[]
        for Rk in s.R:
            c=np.asarray(Rk.sum(1)).ravel(); m=c>0
            nb=(Rk@r)/np.maximum(c,1)
            out.append(float(np.corrcoef(r[m],nb[m])[0,1]))
        return out
    def topk(s,score,p,target):
        o=np.argsort(-score); cl=np.cumsum(s.D.length[o])/s.D.length.sum(); k=np.searchsorted(cl,p)+1
        return float(target[o[:k]].sum()/target.sum()), o[:k]
    def all(s,P,name=''):
        res={}
        res['dev']=float(s._dev(P)); res['skill']=1-res['dev']/s.devHA
        Yh=np.stack([P[t] for t in TE],1)
        res['mae']=float(np.abs(Yh-s.Y).mean()); res['rmse']=float(np.sqrt(((Yh-s.Y)**2).mean()))
        res['calib']=float(Yh.sum()/s.Y.sum())
        r,Ya=s.resid(P)
        for p in [0.01,0.05]:
            cap,sel=s.topk(Ya,p,s.Ya); res[f'cap{int(p*100)}']=cap; res[f'pai{int(p*100)}']=cap/p
            res[f'ka{int(p*100)}']=s.topk(Ya,p,s.KAa)[0]; res[f'rural_sel{int(p*100)}']=float((~s.D.urban[sel]).mean())
        from scipy.stats import spearmanr
        res['spearman']=float(spearmanr(Ya,s.Ya).statistic)
        res['moran']=s.moran(r); res['corr']=s.correlogram(r)
        for g,m in s.groups.items():
            dg=s._dev(P,m); dh=s._dev(s.HA,m)
            res[f'skill_{g}']=1-dg/dh; res[f'dev_{g}']=float(dg); res[f'calib_{g}']=float(Yh[m].sum()/s.Y[m].sum())
        return res

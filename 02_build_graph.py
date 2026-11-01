import pandas as pd, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
W=float(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 0.5
tag=str(W).replace('.','p')
import os
U=os.path.join(os.environ.get('HSIS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','hsis_nc')),'')
C=pd.concat([pd.read_csv(f'{U}nc{y}crash.csv',low_memory=False,usecols=['ROUTE_ID','MILEPOST','TO_RD','REFDISMI','COUNTY']) for y in [21,22,23,24]])
C=C[C.ROUTE_ID.notna()&(C.MILEPOST<999)&(C.REFDISMI<=0.1)]
C['other']=pd.to_numeric(C.TO_RD,errors='coerce')*1000+C.COUNTY; C=C[C.other.notna()]
C['route']=C.ROUTE_ID.astype(np.int64); C['other']=C.other.astype(np.int64)
B=pd.read_parquet(f'bins_{tag}.parquet',columns=['ROUTE_ID','bin','year'])
keys=B[B.year==2024][['ROUTE_ID','bin']].sort_values(['ROUTE_ID','bin']).reset_index(drop=True); keys['node']=np.arange(len(keys))
routes=set(keys.ROUTE_ID); node=keys.set_index(['ROUTE_ID','bin']).node
C=C[C.route.isin(routes)&C.other.isin(routes)&(C.route!=C.other)]
C['bin']=np.floor(C.MILEPOST/W).astype(int)
C=C[pd.MultiIndex.from_arrays([C.route,C.bin]).isin(node.index)]
# evidence: (route,bin) touches other route; junction bin on A = modal bin per (route,other) cluster
C=C.sort_values(['route','other','MILEPOST'])
C['grp']=(C.groupby(['route','other']).MILEPOST.diff().fillna(99)>1.0).cumsum()
J=C.groupby(['route','other','grp']).agg(bin=('bin',lambda s:s.mode().iat[0]),n=('bin','size'),mp=('MILEPOST','median')).reset_index()
nb=keys.groupby('ROUTE_ID').bin.agg(['min','max','size'])
edges=[]; how={'reverse':0,'single_bin':0}
Jr=J.set_index(['route','other'])
for (a,b),g in J.groupby(['route','other']):
    if a>b and (b,a) in Jr.index: continue  # handled from smaller side
    if (b,a) in Jr.index:
        h=Jr.loc[[(b,a)]]
        ga=g.sort_values('mp').bin.values; hb=h.sort_values('mp').bin.values
        if len(ga)==len(hb):
            for x,yb in zip(ga,hb): edges.append((node[(a,x)],node[(b,yb)])); how['reverse']+=1
        else:
            # pair the best-supported cluster on each side
            edges.append((node[(a,g.sort_values('n').bin.values[-1])],node[(b,h.sort_values('n').bin.values[-1])])); how['reverse']+=1
    elif nb.loc[b,'size']==1:
        for x in g.bin.values: edges.append((node[(a,x)],node[(b,nb.loc[b,'min'])])); how['single_bin']+=1
# also: if a is single-bin and b evidence absent handled when iterating (b,a)? add symmetric single-bin case
for (a,b),g in J.groupby(['route','other']):
    if (b,a) not in Jr.index and nb.loc[a,'size']==1 and nb.loc[b,'size']>1:
        pass  # position on b unknown
e_j=np.array(edges)
r=keys.ROUTE_ID.values; bb=keys.bin.values; n=keys.node.values
m=(r[1:]==r[:-1])&(bb[1:]-bb[:-1]==1); e_r=np.c_[n[:-1][m],n[1:][m]]
E=np.r_[e_r,e_j]; E=np.sort(E,1); E=E[E[:,0]!=E[:,1]]; E,idx=np.unique(E,axis=0,return_index=True)
et=np.r_[np.zeros(len(e_r)),np.ones(len(e_j))][idx] if False else None
isj=np.r_[np.zeros(len(e_r),bool),np.ones(len(e_j),bool)]
Eall=np.sort(np.r_[e_r,e_j],1); jset=set(map(tuple,np.sort(e_j,1)))
et=np.array([ (a,b) in jset for a,b in E]).astype(int)
np.save(f'edges_{tag}.npy',E); np.save(f'etype_{tag}.npy',et); keys.to_parquet(f'nodes_{tag}.parquet')
N=len(keys); deg=np.bincount(E.ravel(),minlength=N)
A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(N,N)); nc,lab=connected_components(A,directed=False); sz=np.bincount(lab)
print(how,'nodes',N,'edges',len(E),'junction',et.sum(),'mean deg',round(deg.mean(),3),'isolated',(deg==0).sum(),'comp',nc,'largest',sz.max(),round(sz.max()/N,3),'nodes in comps>=100',round(sz[lab][sz[lab]>=100].size/N,3))

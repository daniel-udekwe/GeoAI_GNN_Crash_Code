import pandas as pd, numpy as np, sys
tag=sys.argv[1] if len(sys.argv)>1 else '0p5'
B=pd.read_parquet(f'bins_{tag}.parquet'); C=pd.read_parquet(f'crash_{tag}.parquet')
keys=pd.read_parquet(f'nodes_{tag}.parquet'); E=np.load(f'edges_{tag}.npy'); et=np.load(f'etype_{tag}.npy')
N=len(keys); T=16
node=keys.set_index(['ROUTE_ID','bin']).node
M=C[C.matched].copy(); M['ROUTE_ID']=M.ROUTE_ID.astype(np.int64)
M['node']=node.reindex(pd.MultiIndex.from_arrays([M.ROUTE_ID,M.bin])).values
M=M[M.node.notna()]; M['node']=M.node.astype(int)
O=np.zeros((N,T),np.float32); np.add.at(O,(M.node.values,M.q.values),1)
KA=np.zeros((N,T),np.float32); ka=M[M.SEVERITY.isin([1,2])]; np.add.at(KA,(ka.node.values,ka.q.values),1)
B=B.merge(keys,on=['ROUTE_ID','bin']).sort_values(['node','year'])
def piv(col): return B.pivot(index='node',columns='year',values=col).reindex(range(N)).values.astype(np.float32)
aadt=piv('aadt'); vmt=piv('vmt'); obs=piv('aadt_obs')
b24=B[B.year==2024].set_index('node').sort_index()
classes=sorted(b24.rdclass.unique()); cls=np.stack([(b24.rdclass==c).values for c in classes],1).astype(np.float32)
static=np.c_[np.log(b24.length.values),b24.lanes.clip(upper=8).values,b24.speed.values/10,b24.townshare.values,cls].astype(np.float32)
deg=np.bincount(E.ravel(),minlength=N); jdeg=np.bincount(E[et==1].ravel(),minlength=N)
np.savez_compressed(f'data_{tag}.npz',O=O,KA=KA,aadt=aadt,vmt=vmt,obs=obs,static=static,classes=np.array(classes),deg=deg,jdeg=jdeg,
  urban=b24.rdclass.str.startswith('Urban').values,county=b24.county.values,length=b24.length.values,rclass=b24.rclass.values,E=E,et=et,
  route=keys.ROUTE_ID.values)
print(tag,N,O.sum(),KA.sum(),O.sum(0))
# plausibility: crash density at junction bins vs not
d=O.sum(1)/b24.length.values
print('crashes/mile junction',d[jdeg>0].mean(),'non-junction',d[jdeg==0].mean(), 'median',np.median(d[jdeg>0]),np.median(d[jdeg==0]))

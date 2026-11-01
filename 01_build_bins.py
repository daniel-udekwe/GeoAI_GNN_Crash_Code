import pandas as pd, numpy as np, sys
W=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
import os
U=os.path.join(os.environ.get('HSIS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','hsis_nc')),'')
bins=[]; crashes=[]
for y in [21,22,23,24]:
    r=pd.read_csv(f'{U}nc{y}road.csv',low_memory=False,usecols=['ROUTE_ID','BEGMP','ENDMP','County','RouteClass','MPLength','FuncClass','SpeedLimit','AADT','ThruLaneCount','RODWYCLS','TownName','TravelDirection'])
    r=r[r.RouteClass.isin([1,2,3,4]) & r.RODWYCLS.notna() & (r.RODWYCLS!='Other') & (r.ENDMP>r.BEGMP)].copy()
    b0=np.floor(r.BEGMP/W).astype(int); b1=np.floor((r.ENDMP-1e-9)/W).astype(int)
    n=(b1-b0+1).values
    rr=r.loc[r.index.repeat(n)].copy()
    rr['bin']=np.concatenate([np.arange(a,a+k) for a,k in zip(b0.values,n)])
    rr['len']=np.minimum(rr.ENDMP,(rr.bin+1)*W)-np.maximum(rr.BEGMP,rr.bin*W)
    rr=rr[rr.len>0]
    rr['has']=rr.AADT.notna(); rr['aL']=rr.AADT.fillna(0)*rr.len; rr['Lh']=rr.len*rr.has
    rr['sL']=rr.SpeedLimit*rr.len; rr['nL']=rr.ThruLaneCount.fillna(2)*rr.len
    rr['town']=rr.TownName.notna()*rr.len
    g=rr.groupby(['ROUTE_ID','bin'])
    a=g.agg(length=('len','sum'),aL=('aL','sum'),Lh=('Lh','sum'),sL=('sL','sum'),nL=('nL','sum'),town=('town','sum'),
            county=('County','first'),rclass=('RouteClass','first'),fclass=('FuncClass','first'),oneway=('TravelDirection',lambda s:(s=='One-way').mean()))
    cls=rr.groupby(['ROUTE_ID','bin','RODWYCLS']).len.sum().reset_index().sort_values('len').drop_duplicates(['ROUTE_ID','bin'],keep='last').set_index(['ROUTE_ID','bin']).RODWYCLS
    a['rdclass']=cls
    a['aadt']=np.where(a.Lh>0,a.aL/a.Lh.where(a.Lh>0,1),np.nan)
    a['speed']=a.sL/a.length; a['lanes']=a.nL/a.length; a['townshare']=a.town/a.length
    a['year']=2000+y
    bins.append(a.drop(columns=['aL','Lh','sL','nL','town']).reset_index())
    c=pd.read_csv(f'{U}nc{y}crash.csv',low_memory=False,usecols=['CASENO','SEVERITY','ALCFLAG','RURURB','COUNTY','CITY','ROUTE_ID','MILEPOST','CRASH_DATE','TIME','ACCTYPE','LIGHT','RDSURF','WEATHER1','NUM_UNIT','WZ_ACT'])
    c['year']=2000+y
    crashes.append(c)
B=pd.concat(bins); C=pd.concat(crashes)
C['date']=pd.to_datetime(C.CRASH_DATE,format='%m/%d/%Y')
C['hour']=pd.to_numeric(C.TIME.str[:2],errors='coerce')
C['dow']=C.date.dt.dayofweek
C['q']=(C.year-2021)*4+C.date.dt.quarter-1
C['ok']=C.ROUTE_ID.notna()&(C.MILEPOST<999)
C['bin']=np.where(C.ok,np.floor(C.MILEPOST.fillna(0)/W),-1).astype(int)
# panel: bins present in all 4 years
cnt=B.groupby(['ROUTE_ID','bin']).year.nunique()
keep=cnt[cnt==4].index
B=B.set_index(['ROUTE_ID','bin']).loc[keep].reset_index()
# AADT imputation: within bin across years, then route, then class-county median
B['aadt_obs']=B.aadt.notna()
B['aadt']=B.groupby(['ROUTE_ID','bin']).aadt.transform(lambda s:s.ffill().bfill())
B['aadt']=B.groupby(['ROUTE_ID','year']).aadt.transform(lambda s:s.interpolate(limit_direction='both'))
B['aadt']=B.aadt.fillna(B.groupby(['rdclass','county','year']).aadt.transform('median')).fillna(B.groupby(['rdclass','year']).aadt.transform('median'))
B['vmt']=B.aadt*B.length*365/1e6  # million VMT per year
C['matched']=C.set_index(['ROUTE_ID','bin']).index.isin(keep) & C.ok
tag=str(W).replace('.','p')
B.to_parquet(f'bins_{tag}.parquet'); C.to_parquet(f'crash_{tag}.parquet')
print(W,'bins',len(keep),'miles',B[B.year==2024].length.sum().round(0),'crashes',len(C),'matched',C.matched.mean().round(4),'aadt observed share (len)',(B.aadt_obs*B.length).sum()/B.length.sum())
print(B[B.year==2024].groupby('rdclass').agg(bins=('bin','size'),miles=('length','sum'),aadt=('aadt','median')))

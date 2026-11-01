import numpy as np, json, time, sys, torch
import gm, ev
from gm import TE
D=gm.Data('0p5'); Ev=ev.Evaluator(D); R={}
def run(name,D_,kind='stgnn',nb=True,off=True,K=2,seeds=(0,)):
    Ps=[]; t0=time.time()
    for sd in seeds:
        m,P,ep=gm.train(D_,kind,seed=sd,epochs=60,lr=1e-2,nb=nb,offset=off,K=K,patience=8); Ps.append(P)
        R.setdefault(name+'_seeds',[]).append(Ev.all(P)['dev'])
    P={t:np.mean([p[t] for p in Ps],0) for t in range(4,16)}
    r=Ev.all(P); r['secs']=round(time.time()-t0); R[name]=r
    print(name,round(r['dev'],4),round(r['skill'],4),round(r['moran'],4),round(r['pai1'],2),flush=True)
    json.dump(R,open('res_abl.json','w'),indent=1); return m
run('no offset',D,off=False)
run('Poisson loss',D,nb=False)
run('no graph (EI-GRU)',D,kind='gru')
rng=np.random.default_rng(42); E=D.E.copy(); E2=np.c_[E[:,0],rng.permutation(E[:,1])]; E2=E2[E2[:,0]!=E2[:,1]]
run('shuffled graph',gm.Data('0p5',edges=E2))
d=np.load('data_0p5.npz'); run('route-only graph',gm.Data('0p5',edges=d['E'][d['et']==0]))
for K in [1,3,4]:
    m=run(f'K={K}',D,K=K)
    if K==4: torch.save(m.state_dict(),'ei_K4.pt')
print('done')

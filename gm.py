import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, math
torch.set_num_threads(2)
TR=list(range(4,10)); VA=[10,11]; TE=[12,13,14,15]
class Data:
    def __init__(s,tag='0p5',edges=None):
        d=np.load(f'data_{tag}.npz',allow_pickle=True); s.d=d; s.tag=tag
        s.O=d['O']; s.KA=d['KA']; s.N=s.O.shape[0]; s.E=d['E'] if edges is None else edges
        s.urban=d['urban']; s.length=d['length']; s.county=d['county']; s.obs=d['obs']; s.et=d['et']
        st=d['static']; deg=np.bincount(s.E.ravel(),minlength=s.N); jdeg=d['jdeg']
        s.X={}; s.off={}
        lo=np.log1p(s.O)
        for t in range(4,16):
            y=t//4; q=t%4
            lags=lo[:,t-4:t][:,::-1]  # t-1..t-4
            ka=np.log1p(s.KA[:,t-4:t].sum(1,keepdims=True))
            qo=np.zeros((s.N,4),np.float32); qo[:,q]=1
            la=np.log(np.clip(d['aadt'][:,y],10,None))[:,None]; ob=d['obs'][:,y][:,None]
            lv=np.log(d['vmt'][:,y]/4+1e-4)[:,None]
            s.X[t]=np.c_[lags,ka,qo,la,ob,lv,st,np.log1p(deg)[:,None],np.log1p(jdeg)[:,None]].astype(np.float32)
            s.off[t]=lv[:,0].astype(np.float32)
        Xtr=np.concatenate([s.X[t] for t in TR]); s.mu=Xtr.mean(0); s.sd=Xtr.std(0)+1e-6
        s.mu[5:9]=0; s.sd[5:9]=1  # keep one-hot unscaled
        s.F=Xtr.shape[1]; s.nlag=4
        # spatial lag features (row-normalised neighbour mean)
        import scipy.sparse as sp
        A=sp.coo_matrix((np.ones(len(s.E)),(s.E[:,0],s.E[:,1])),shape=(s.N,s.N)); A=(A+A.T).tocsr(); A.data[:]=1
        dg=np.asarray(A.sum(1)).ravel(); s.Arow=sp.diags(1/np.maximum(dg,1))@A; s.A=A; s.dg=dg
        s.Xs={}
        for t in range(4,16):
            l1=s.Arow@lo[:,t-4:t][:,::-1]; l2=s.Arow@(s.Arow@lo[:,t-4:t].sum(1)); 
            s.Xs[t]=np.c_[l1,l2,(dg>0)].astype(np.float32)
        # torch adjacency (sym norm with self loops)
        Ah=A+sp.eye(s.N); dd=np.asarray(Ah.sum(1)).ravel(); Dm=sp.diags(dd**-0.5); An=(Dm@Ah@Dm).tocoo()
        s.Ahat=torch.sparse_coo_tensor(np.vstack([An.row,An.col]),An.data.astype(np.float32),(s.N,s.N)).coalesce()
        ei=np.vstack([np.r_[s.E[:,0],s.E[:,1],np.arange(s.N)],np.r_[s.E[:,1],s.E[:,0],np.arange(s.N)]])
        s.ei=torch.tensor(ei,dtype=torch.long)  # src,dst incl self loops
        s.etype=torch.tensor(np.r_[s.et,s.et,np.full(s.N,2)],dtype=torch.long)
    def x(s,t): return torch.tensor((s.X[t]-s.mu)/s.sd)
    def seq(s,t):  # N x 4 x 2 : log1p count, qoy index scaled, oldest->newest
        lo=np.log1p(s.O[:,t-4:t]); qs=np.tile(((np.arange(t-4,t)%4)/3.0)[None,:],(s.N,1))
        return torch.tensor(np.stack([lo,qs],2).astype(np.float32))
    def y(s,t): return torch.tensor(s.O[:,t])
class GCNLayer(nn.Module):
    def __init__(s,i,o): super().__init__(); s.l=nn.Linear(i,o)
    def forward(s,x,A): return torch.sparse.mm(A,s.l(x))
class GATLayer(nn.Module):
    def __init__(s,i,o,heads=4,concat=True):
        super().__init__(); s.h=heads; s.o=o; s.l=nn.Linear(i,o*heads,bias=False); s.a_s=nn.Parameter(torch.randn(heads,o)*0.1); s.a_d=nn.Parameter(torch.randn(heads,o)*0.1); s.concat=concat; s.b=nn.Parameter(torch.zeros(o*heads if concat else o))
        s.last=None
    def forward(s,x,ei):
        N=x.shape[0]; h=s.l(x).view(N,s.h,s.o); src,dst=ei[0],ei[1]
        e=F.leaky_relu((h[src]*s.a_s).sum(-1)+(h[dst]*s.a_d).sum(-1),0.2)  # E x H
        m=torch.full((N,s.h),-1e9).scatter_reduce(0,dst[:,None].expand(-1,s.h),e,'amax',include_self=True)
        w=torch.exp(e-m[dst]); den=torch.zeros(N,s.h).index_add_(0,dst,w); a=w/(den[dst]+1e-16)
        s.last=a.detach()
        out=torch.zeros(N,s.h,s.o).index_add_(0,dst,a[:,:,None]*h[src])
        out=out.reshape(N,-1) if s.concat else out.mean(1)
        return out+s.b
class Net(nn.Module):
    """kind: mlp, gcn, gat, gru, stgnn ; layers K for graph"""
    def __init__(s,kind,F_,hid=64,K=2,nb=False,offset=False,static_dim=None):
        super().__init__(); s.kind=kind; s.K=K; s.nb=nb; s.offset=offset
        s.inp=nn.Linear(F_,hid)
        if kind in ('gru','stgnn'):
            s.gru=nn.GRU(2,32,batch_first=True); s.inp=nn.Linear(F_+32,hid)
        if kind in ('gcn','stgnn'): s.layers=nn.ModuleList([GCNLayer(hid,hid) for _ in range(K)])
        if kind=='gat': s.layers=nn.ModuleList([GATLayer(hid,hid//4,4) for _ in range(K)])
        if kind in ('mlp','gru'): s.layers=nn.ModuleList([nn.Linear(hid,hid) for _ in range(K)])
        s.out=nn.Linear(hid,1); s.logalpha=nn.Parameter(torch.tensor(-0.5))
    def forward(s,x,seq,off,D):
        if s.kind in ('gru','stgnn'):
            _,h=s.gru(seq); x=torch.cat([x,h[0]],1)
        h=F.relu(s.inp(x))
        for L in s.layers:
            if s.kind in ('gcn','stgnn'): h=F.relu(L(h,D.Ahat))+h
            elif s.kind=='gat': h=F.elu(L(h,D.ei))+h
            else: h=F.relu(L(h))+h
        eta=s.out(h)[:,0]
        if s.offset: eta=eta+off
        return eta
def nll(eta,y,model):
    if model.nb:
        a=torch.exp(model.logalpha); mu=torch.exp(eta); r=1/a
        ll=torch.lgamma(y+r)-torch.lgamma(r)-torch.lgamma(y+1)+r*torch.log(r/(r+mu))+y*torch.log(mu/(r+mu)+1e-12)
        return -ll
    return torch.exp(eta)-y*eta
def train(D,kind,seed=0,epochs=80,lr=3e-3,K=2,nb=False,offset=False,mask=None,patience=10,hid=64,verbose=False,extra_off=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m=Net(kind,D.F,hid,K,nb,offset); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-5)
    cache={t:(D.x(t),D.seq(t),torch.tensor(D.off[t] if extra_off is None else extra_off[t]),D.y(t)) for t in TR+VA+TE}
    msk=torch.ones(D.N,dtype=torch.bool) if mask is None else torch.tensor(mask)
    best=1e18; bs=None; bad=0; t0=time.time()
    for ep in range(epochs):
        m.train()
        for t in np.random.permutation(TR):
            x,sq,off,y=cache[t]; opt.zero_grad(); eta=m(x,sq,off,D); loss=nll(eta,y,m)[msk].mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v=sum(dev(torch.exp(m(*cache[t][:3],D)),cache[t][3])[msk].mean().item() for t in VA)/len(VA)
        if v<best-1e-5: best=v; bs={k:v_.clone() for k,v_ in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
        if verbose: print(ep,round(v,5),round(time.time()-t0,1),flush=True)
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): P={t:torch.exp(m(*cache[t][:3],D)).numpy() for t in range(4,16)}
    return m,P,ep+1
def dev(mu,y):
    if isinstance(mu,np.ndarray):
        return 2*(np.where(y>0,y*np.log(np.where(y>0,y,1)/mu),0)-(y-mu))
    return 2*(torch.where(y>0,y*torch.log(torch.where(y>0,y,torch.ones_like(y))/mu),torch.zeros_like(y))-(y-mu))

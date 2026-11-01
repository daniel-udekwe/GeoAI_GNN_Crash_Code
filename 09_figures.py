import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'legend.frameon':False,'pdf.fonttype':42})
G=''  # run from the working directory that holds the experiment outputs
M=json.load(open(G+'res_main_0p5.json'))
order=['HA','NB-SPF','NB-lag','LGBM','LGBM-S','MLP','GRU','GCN','GAT','ST-GNN','EI-STGNN']
fam={'HA':'b','NB-SPF':'s','NB-lag':'s','LGBM':'t','LGBM-S':'t','MLP':'n','GRU':'n','GCN':'g','GAT':'g','ST-GNN':'g','EI-STGNN':'e'}
col={'b':'#bdbdbd','s':'#969696','t':'#737373','n':'#525252','g':'#6baed6','e':'#08519c'}
W=6.3
import os; os.makedirs('figures',exist_ok=True)
order=[m for m in order if m in M]
# Fig skill (zoomed dot plot of learned models)
dh=M['HA']['dev']
lm=['LGBM','LGBM-S','MLP','GRU','GCN','GAT','ST-GNN','EI-STGNN']
fig,ax=plt.subplots(figsize=(W,2.5))
for i,m in enumerate(lm):
    s=M.get(m+'_seeds')
    if s and len(s)>1:
        ss=[1-d/dh for d in s]; ax.plot([min(ss),max(ss)],[i,i],color=col[fam[m]],lw=2.2,alpha=0.45,solid_capstyle='round')
    ax.plot(M[m]['skill'],i,'o',color=col[fam[m]],ms=6)
    ax.text(M[m]['skill']+0.0004,i-0.28,f"{M[m]['skill']:.4f}",fontsize=7)
ax.axvline(M['LGBM-S']['skill'],color='#737373',lw=0.7,ls=':')
ax.set_yticks(range(len(lm))); ax.set_yticklabels(lm); ax.invert_yaxis()
ax.set_xlabel('Skill vs. seasonal historical average, 2024 (dot: seed ensemble; bar: range of single seeds)')
ax.set_xlim(0.398,0.412)
fig.tight_layout(); fig.savefig('figures/fig_skill.pdf'); plt.close()
# Fig correlogram + influence
D=json.load(open(G+'diag.json')) if __import__('os').path.exists(G+'diag.json') else {}
fig,axs=plt.subplots(1,2,figsize=(W,2.5),gridspec_kw={'wspace':0.75})
ax=axs[0]
for m,ls in [('NB-SPF','--'),('NB-lag','-'),('LGBM','-'),('LGBM-S','-'),('MLP','-'),('GCN','-'),('EI-STGNN','-')]:
    if m in M: ax.plot([1,2,3,4],M[m]['corr'],marker='o',ms=3,ls=ls,color=col[fam[m]],lw=1.6 if m=='EI-STGNN' else 1,label=m)
ax.axhline(0,color='k',lw=0.5); ax.set_xticks([1,2,3,4]); ax.set_xlabel('Network distance (hops)'); ax.set_ylabel('Residual correlation'); ax.set_title('(a) Residual network correlogram, 2024',fontsize=8.5,loc='left'); ax.legend(fontsize=6.3,ncol=1,loc='center left',bbox_to_anchor=(1.0,0.5))
ax=axs[1]
for k,(name,c) in enumerate([('K=2','#6baed6'),('K=4','#08519c')]):
    v=D.get(f'influence_{name}')
    if v: ax.bar(np.arange(6)+(k-0.5)*0.38,np.array(v)*100,width=0.38,color=c,label=f'EI-STGNN, {name}')
ax.set_xticks(range(6)); ax.set_xticklabels(['0\n(self)','1','2','3','4','>4']); ax.set_xlabel('Network distance (hops)'); ax.set_ylabel('Share of input gradient (%)'); ax.set_title('(b) Learned influence',fontsize=8.5,loc='left'); ax.legend(fontsize=6.5)
fig.subplots_adjust(left=0.08,right=0.98,bottom=0.22,top=0.88); fig.savefig('figures/fig_dependence.pdf'); plt.close()
# Fig groups
fig,ax=plt.subplots(figsize=(W,2.5))
grp=[('urban','Urban'),('rural','Rural'),('aadt_obs','AADT measured'),('aadt_imp','AADT imputed')]
ms=[m for m in order if m!='HA']; x=np.arange(len(ms)); w=0.2
for k,(g,lab) in enumerate(grp):
    ax.bar(x+(k-1.5)*w,[M[m][f'skill_{g}'] for m in ms],width=w,label=lab,color=['#9ecae1','#3182bd','#fdae6b','#e6550d'][k])
ax.set_xticks(x); ax.set_xticklabels(ms,rotation=30,ha='right'); ax.set_ylabel('Within-group skill vs. HA'); ax.axhline(0,color='k',lw=0.5); ax.legend(ncol=4,fontsize=7,loc='lower center',bbox_to_anchor=(0.5,1.0)); ax.set_ylim(-0.08,0.5)
fig.tight_layout(); fig.savefig('figures/fig_groups.pdf'); plt.close()
print('figs ok')
# Fig CV
import os
if os.path.exists(G+'res_cv.json'):
    R=json.load(open(G+'res_cv.json'))
    ms=['LGBM','LGBM-S','MLP','GCN']
    fig,ax=plt.subplots(figsize=(W,2.3))
    for i,m in enumerate(ms):
        for j,(sch,c,mk) in enumerate([('random','#9ecae1','o'),('county-blocked','#08519c','s')]):
            f=R[f'{sch}|{m}']['folds']; x=i+(j-0.5)*0.32
            ax.scatter([x]*len(f),f,color=c,marker=mk,s=16,zorder=3,label=('Random node folds' if sch=='random' else 'County-blocked folds') if i==0 else None)
            ax.plot([x-0.11,x+0.11],[np.mean(f)]*2,color='k',lw=1.2)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylabel('Skill on held-out nodes, 2024')
    ax.legend(fontsize=7,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2)
    fig.tight_layout(); fig.savefig('figures/fig_cv.pdf'); plt.close()
print('cv fig ok')
# Fig scale
if os.path.exists(G+'res_main_0p25.json') and 'EI-STGNN' in json.load(open(G+'res_main_0p25.json')):
    S={t:json.load(open(G+f'res_main_{t}.json')) for t in ['0p25','0p5','1p0']}
    ms=['NB-lag','LGBM-S','MLP','GCN','EI-STGNN']; xs=[0.25,0.5,1.0]
    fig,axs=plt.subplots(1,2,figsize=(W,2.4))
    for m in ms:
        c=col[fam[m]]; lw=1.8 if m in ('EI-STGNN','LGBM-S') else 1.1
        axs[0].plot(xs,[S[t][m]['skill'] for t in S],marker='o',ms=3.5,color=c,lw=lw,label=m)
        axs[1].plot(xs,[S[t][m]['moran'] for t in S],marker='o',ms=3.5,color=c,lw=lw,label=m)
    for ax,yl,tt in [(axs[0],'Skill vs. HA','(a) Forecast skill'),(axs[1],"Residual Moran's I",'(b) Residual network dependence')]:
        ax.set_xscale('log'); ax.set_xticks(xs); ax.set_xticklabels(['0.25','0.5','1.0']); ax.minorticks_off(); ax.set_xlabel('Bin length (miles)'); ax.set_ylabel(yl); ax.set_title(tt,fontsize=8.5,loc='left')
    axs[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig('figures/fig_scale.pdf'); plt.close(); print('scale fig ok')

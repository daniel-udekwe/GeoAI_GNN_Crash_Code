import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
import os, urllib.request
if not os.path.exists('nc.json'):  # North Carolina county boundaries (FIPS 37xxx) from plotly/datasets
    url='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    allc=json.load(urllib.request.urlopen(url))
    json.dump({'type':'FeatureCollection','features':[f for f in allc['features'] if f['id'].startswith('37')]},open('nc.json','w'))
os.makedirs('figures',exist_ok=True)
gj=json.load(open('nc.json')); g=pd.read_csv('county_metrics.csv',dtype={'fips':str}).set_index('fips')
k=np.cos(np.radians(35.5))
def polys(f):
    geo=f['geometry']; rings=geo['coordinates'] if geo['type']=='Polygon' else [p[0] for p in geo['coordinates']]
    if geo['type']=='Polygon': rings=[rings[0]]
    return [np.array(r)*[k,1] for r in rings]
def draw(ax,vals,cmap,norm,title,cbl,ticks=None,edge='white'):
    patches=[];v=[]
    for f in gj['features']:
        for p in polys(f): patches.append(Polygon(p,closed=True)); v.append(vals.get(f['id'],np.nan))
    pc=PatchCollection(patches,cmap=cmap,norm=norm,edgecolor=edge,linewidth=0.4); pc.set_array(np.array(v)); ax.add_collection(pc)
    ax.set_xlim(-84.4*k,-75.4*k); ax.set_ylim(33.75,36.65); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title,fontsize=9.5,loc='left',pad=3)
    cax=ax.inset_axes([1.01,0.12,0.018,0.76]); cb=plt.colorbar(pc,cax=cax,orientation='vertical'); cb.set_label(cbl,fontsize=8,labelpad=3); cb.ax.tick_params(labelsize=7.5,length=2)
    if ticks is not None: cb.set_ticks(ticks); cb.set_ticklabels([str(t) for t in ticks])
    else:
        from matplotlib.ticker import MaxNLocator; cb.ax.yaxis.set_major_locator(MaxNLocator(4))
    return pc
W=6.3
# Study area
fig,axs=plt.subplots(2,1,figsize=(W,5.0))
draw(axs[0],g.density.to_dict(),'Reds',LogNorm(1,100),'(a) Located crashes per route-mile, 2021–2024','crashes per route-mile (log scale)',ticks=[1,3,10,30,100])
draw(axs[1],(g.imp*100).to_dict(),'Purples',Normalize(20,80),'(b) Bins with imputed AADT','% of bins in county',ticks=[20,40,60,80])
fig.subplots_adjust(left=0.01,right=0.86,top=0.95,bottom=0.02,hspace=0.18); fig.savefig('figures/fig_studyarea.pdf'); plt.close()
# Results maps
fig,axs=plt.subplots(3,1,figsize=(W,7.4))
draw(axs[0],g['skill_LGBM-S'].to_dict(),'Blues',Normalize(0.25,0.55),'(a) LGBM-S skill by county','skill vs. HA, 2024',ticks=[0.3,0.4,0.5])
lim=0.02; draw(axs[1],g.gain_gcn.to_dict(),'RdBu',TwoSlopeNorm(0,-lim,lim),'(b) Graph gain: GCN − MLP','skill difference',ticks=[-0.02,-0.01,0,0.01,0.02])
draw(axs[2],g.fold_skill_lgbms.to_dict(),'viridis',Normalize(0.37,0.43),'(c) County-blocked CV folds','LGBM-S skill of fold',ticks=[0.37,0.39,0.41,0.43])
fig.subplots_adjust(left=0.01,right=0.86,top=0.97,bottom=0.01,hspace=0.16); fig.savefig('figures/fig_maps.pdf'); plt.close()
# Heatmap class x model
H=pd.read_csv('class_skill.csv'); short={'Rural 2 Lane Roads':'Rural 2-lane','Rural Freeways':'Rural freeway','Rural Multilane Divided Non-Freeway':'Rural multilane div.','Rural Multilane Undivided Non-Freeway':'Rural multilane undiv.','Urban 2 Lane Roads':'Urban 2-lane','Urban Freeways':'Urban freeway','Urban Multilane Divided Non-Freeway':'Urban multilane div.','Urban Multilane Undivided Non-Freeway':'Urban multilane undiv.'}
ms=['LGBM','LGBM-S','MLP','GRU','GCN','GAT','ST-GNN','EI-STGNN']
A=H[ms].values; rel=A-A.max(1,keepdims=True)
fig,ax=plt.subplots(figsize=(W,2.7))
im=ax.imshow(rel*100,cmap='Blues',vmin=-2.5,vmax=0,aspect='auto')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j,i,f'{A[i,j]:.3f}',ha='center',va='center',fontsize=6.8,color='white' if rel[i,j]*100>-1.2 else 'black',fontweight='bold' if rel[i,j]==0 else 'normal')
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms,rotation=25,ha='right',fontsize=7); ax.set_yticks(range(len(H))); ax.set_yticklabels([f"{short[c]} (n={n:,})" for c,n in zip(H['class'],H['n'])],fontsize=7)
cb=plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label('Gap to best model in row (pp, clipped at -2.5)',fontsize=7); cb.ax.tick_params(labelsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig_heatmap.pdf'); plt.close()
print('ok')

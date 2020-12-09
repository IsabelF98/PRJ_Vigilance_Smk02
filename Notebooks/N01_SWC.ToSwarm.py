# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Vigilance
#     language: python
#     name: vigilance
# ---

# # Description: SWC + Embeddings
#
# This notebook performs the following analysis steps:
#
# 1) Load ROI representative time series (those must already exists in text file format)
#
# 2) Plots static FC matrix, as well as a carpet plot
#
# 3) Dimensionality Reduction from ROI to PCA components (whole time-series)
#
# 4) Compute Sliding Window Correlation based on PCA representative time series
#
# 5) Generate 3D Laplacian Embeddings

# %%time
import pickle
import sys
import os.path as osp
import os
import pandas as pd
import xarray as xr
import numpy as np
from nilearn.plotting import plot_matrix
from scipy.signal import tukey, hamming
from sklearn.manifold  import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import correlation as dis_corr
from utils.base import plot_fc_matrix, compute_swc, reduce_dimensionality_pca
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')
pn.extension()

seed = np.random.RandomState(seed=7)

# +
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/valid_run_df.pkl')

# Dictionary of subject with valid runs
SubDict = {}
for i,idx in enumerate(sub_DF.index):
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    time = sub_DF.loc[idx]['Time']
    tp_min = sub_DF.loc[idx]['Time Point Min']
    tp_max = sub_DF.loc[idx]['Time Point Max']
    if sbj in SubDict.keys():
        SubDict[sbj].append((run,time,tp_min,tp_max))
    else:
        SubDict[sbj] = [(run,time,tp_min,tp_max)]
SubjectList = list(SubDict.keys()) # list of subjects        
for sbj in SubjectList:
    SubDict[sbj].append(('All',sum(SubDict[sbj][i][1] for i in range(0,len(SubDict[sbj]))),0,sum(SubDict[sbj][i][1] for i in range(0,len(SubDict[sbj])))-1))

# +
SBJ                    = sys.argv[1]
RUN                    = sys.argv[2]
WL_sec                 = int(sys.argv[3])
TIME                   = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ])) if SubDict[SBJ][i][0] == RUN][0]
tp_min                 = [SubDict[SBJ][i][2] for i in range(0,len(SubDict[SBJ])) if SubDict[SBJ][i][0] == RUN][0]
tp_max                 = [SubDict[SBJ][i][3] for i in range(0,len(SubDict[SBJ])) if SubDict[SBJ][i][0] == RUN][0]
atlas_name             = 'Craddock_T2Level_0200'
TR                     = 2.0
WS_trs                 = 1
WL_trs                 = int(WL_sec / TR)
dim_red_method         = 'PCA'
dim_red_method_percent = 97.5
le_num_dims            = 3
le_k_NN                = 100

path_ts        = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI','errts.'+SBJ+'.'+atlas_name+'.wl'+str(WL_sec).zfill(3)+'s.fanaticor_ts.1D')
path_outdir    = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI')
out_prefix     = SBJ+'_fanaticor_'+atlas_name+'_wl'+str(WL_sec).zfill(3)+'s_ws'+str(int(WS_trs*TR)).zfill(3)+'s_'+RUN
out_pca_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.pca_obj.pkl')
out_pcats_path = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.pca_ts.pkl')
out_swc_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.swcorr.pkl')
out_lem_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.le'+str(le_num_dims)+'d_knn'+str(le_k_NN).zfill(3)+'.pkl')

print('++ INFO: Selection Parameters: ')
print(' + Subject         : %s' % SBJ)
print(' + Run.            : %s' % RUN)
print(' + Atlas           : %s' % atlas_name)
print(' + SWC             : wl=%ss, ws=%ss, dim_red=%s, extra-->%s' % (str(WL_sec),str(WS_trs*TR),dim_red_method,'vk='+str(dim_red_method_percent)+'%'))
print(' + Timeseries File : %s' % path_ts)
print(' + -----------------------------------------------------------')
print('++ INFO: Laplacian Embedding Settings: ')
print(' + Number of Dimensions: %d' % le_num_dims)
print(' + K-Nearest Neighbors : %d' % le_k_NN)
print(' + Distance Metric: correlation distance')
print('++ -----------------------------------------------------------')
print(' + INFO: Outputs:')
print(' + Output Folder       : %s' % path_outdir)
print(' + PCA Object File     : %s' % out_pca_path)
print(' + PCA Timeseries File : %s' % out_pcats_path)
print(' + SWC File            : %s' % out_swc_path)
print(' + LE  File            : %s' % out_lem_path)
# -

# ***
# ### 1. Load ROI Timeseries
#
# First, we load the time series for all representative ROIs, and show a static functional connectivity matrix and a carpet plot. This may help capture some issues with the data.

# +
# %%time

temp_ts_df = pd.read_csv(path_ts, sep='\t', header=None)
if RUN != 'All':
    ts_df = pd.DataFrame(temp_ts_df.loc[tp_min:tp_max])
    ts_df = ts_df.reset_index()
    ts_df = ts_df.drop('index',axis=1)
else:
    ts_df = pd.DataFrame(temp_ts_df)
Nacq,Nrois = ts_df.shape

# Generate ROI names
# ------------------
# Those are default names, but it would be useful to have a file per atlas that contains the names
# and we load it here.
roi_names  = ['ROI'+str(r+1).zfill(3) for r in range(Nrois)]

# Put timeseries also in Xarray form. This is necessary for plotting purposes via hvplot.Image
# --------------------------------------------------------------------------------------------
ts_xr      = xr.DataArray(ts_df.values,dims=['Time [TRs]','ROIs'])

# Show a summary of the data being loaded.
# ----------------------------------------
print('++ INFO: Time-series loaded into memory [N_acq=%d, N_rois=%d]' % (Nacq, Nrois))


# ***
# ### 2. Dimensionality Reduction
#
# Here we reduce the dimensionality of the data via PCA. The goal is to have a smaller connectivity matrix, therefore we go from X number of ROIs to a Y number of PCA components, with Y hopefully being much smaller than X.
#
# * How many components are kept depends on the amount of variance we keep (default is 97.5%) 

# %%time
ts_pca_df, pca_plot, pca = reduce_dimensionality_pca(ts_df,dim_red_method_percent,sbj_id=SBJ)
pickle.dump(pca, open(out_pca_path, "wb" ) )
ts_pca_df.to_pickle(out_pcats_path)


# ***
# ### 4. Create SWC Matrix

# %%time
# Create a tukey (or tappered window) of the appropriate length
# =============================================================
#window = tukey(WL_trs,.2)
window = np.ones((WL_trs,))

# %%time
# Compute sliding window correlation
# ==================================
swc_r, swc_Z, winInfo = compute_swc(ts_pca_df,WL_trs,WS_trs,window=window)
xr.DataArray(swc_Z.values.T,dims=['Time [Window ID]','PCA Connection'])

# ***
# ### 4. Generate Laplacian Embedding

# %%time
se             = SpectralEmbedding(n_components=le_num_dims, affinity='precomputed', n_jobs=32, random_state=seed)
X_affinity     = kneighbors_graph(swc_Z.T,le_k_NN,include_self=True,n_jobs=32, metric=dis_corr)
X_affinity     = 0.5 * (X_affinity + X_affinity.T)
se_X           = se.fit_transform(X_affinity.toarray())
print ('++ INFO: Embedding Dimensions: %s' % str(se_X.shape))

# +
# Put the embeddings into a dataframe (for saving and plotting)
# =============================================================
LE3D_df      = pd.DataFrame(columns=['x','y','z','x_norm','y_norm','z_norm','no_color_rgb','no_color_hex','time_color_rgb','label'])
LE3D_df['x'] = se_X[:,0]
LE3D_df['y'] = se_X[:,1]
LE3D_df['z'] = se_X[:,2]
# Note: there is a change in scale between scikit-learn 0.19 and 0.23 when it comes to the laplacian embeddings.
# I checked a few examples and the structure is the same, but the scale is different. To be able to represent all cases
# on the same scale (and given that the dimensions are meaningless), I create this normalized version of the low dimensional embedding
LE3D_df[['x_norm','y_norm','z_norm']]= LE3D_df[['x','y','z']]/LE3D_df[['x','y','z']].max()
# External-data based color
LE3D_df['no_color_rgb'] = [(204,209,209) for i in range(winInfo['numWins'])]
LE3D_df['no_color_hex'] = ['#CCD1D1' for i in range(winInfo['numWins'])]

# Time-based color
time_color_rbg_temp = pd.DataFrame(LE3D_df['time_color_rgb'])
if RUN == 'All':
    time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SubjSelect.value])-1)]
    color_list = [(255,87,34),(255,167,38),(255,235,59),(139,195,74),(0,188,212),(126,87,194)]
    x=0
    for i in range(len(time_list)):
        time_color_rbg_temp.loc[x:(x-1)+time_list[i]-(WL_trs-1), 'time_color_rgb'] = [color_list[i]] # color for run windows
        x=x+time_list[i]-(WL_trs-1)
        if i != len(time_list)-1:
            time_color_rbg_temp.loc[x:(x-1)+(WL_trs-1), 'time_color_rgb'] = [(204,209,209)] # color for between run windows
            x=x+(WL_trs-1)
    LE3D_df['time_color_rgb'] = time_color_rbg_temp['time_color_rgb']
else:
    time_color_rbg_temp.loc[0:244, 'time_color_rgb'] = [(n,0,0) for n in range(10,255)]
    time_color_rbg_temp.loc[245:winInfo['numWins']-1, 'time_color_rgb'] = [(255,n,n) for n in range(winInfo['numWins']-245)]
    LE3D_df['time_color_rgb'] = time_color_rbg_temp['time_color_rgb']

# Window Names
LE3D_df['label'] = winInfo['winNames']
LE3D_df.head()
LE3D_df.to_pickle(out_lem_path)
# -

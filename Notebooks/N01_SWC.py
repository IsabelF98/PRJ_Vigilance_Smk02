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
import os.path as osp
import os
import pandas as pd
import xarray as xr
import numpy as np
from statistics import mean
from nilearn.plotting import plot_matrix
from scipy.signal import tukey, hamming
from sklearn.manifold  import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import correlation as dis_corr
from utils.base import plot_fc_matrix, compute_swc, reduce_dimensionality_pca
from utils.emb_func import load_data, lapacian_dataframe
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')
pn.extension()

# ***
# ### 0. Load Preliminary Information

seed = np.random.RandomState(seed=7)

# +
# Load all subject and run information for valid runs
# ---------------------------------------------------

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# Load data frame of valid subjects info
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl')

# Dictionary of subject with valid runs
# The dictionary is organized by subject. Keys are the subject and the values are a list of tuples as such:
# (run name, number of TR's in the data, min index of run in the concatinated data, max index of run in the concatinated data)
SubDict = {} # Empty dictionary
for i,idx in enumerate(sub_DF.index): # Iterate through each row of data frame
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    time = sub_DF.loc[idx]['Time']
    tp_min = sub_DF.loc[idx]['Time Point Min']
    tp_max = sub_DF.loc[idx]['Time Point Max']
    if sbj in SubDict.keys():
        SubDict[sbj].append((run,time,tp_min,tp_max)) # Add run tuple (described above)
    else:
        SubDict[sbj] = [(run,time,tp_min,tp_max)] # If subject is not already in the directory a new element is created
SubjectList = list(SubDict.keys()) # list of subjects        
# Add 'All' option to subject diction for each subject. 'All' meaning the concatinated data 
for sbj in SubjectList:
    SubDict[sbj].append(('All',sum(SubDict[sbj][i][1] for i in range(0,len(SubDict[sbj]))),0,sum(SubDict[sbj][i][1] for i in range(0,len(SubDict[sbj])))-1))

# +
# Widgets for selecting subject run and window legth
# --------------------------------------------------
SubjSelect = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0]) # Select subject
RunSelect  = pn.widgets.Select(name='Select Run', options=[SubDict[SubjSelect.value][i][0] for i in range(0,len(SubDict[SubjSelect.value]))]) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length', options=[30,46,60]) # Select window lenght

# Updates available runs given SubjSelect value
def update_run(event):
    RunSelect.options = [SubDict[event.new][i][0] for i in range(0,len(SubDict[event.new]))]
SubjSelect.param.watch(update_run, 'value')

# Displays the three widgets created above
pn.Row(SubjSelect, RunSelect, WindowSelect)

# +
# Variables for running embeddings
SBJ                    = SubjSelect.value
RUN                    = RunSelect.value
TIME                   = [SubDict[SubjSelect.value][i][1] for i in range(0,len(SubDict[SubjSelect.value])) if SubDict[SubjSelect.value][i][0] == RunSelect.value][0]
tp_min                 = [SubDict[SubjSelect.value][i][2] for i in range(0,len(SubDict[SubjSelect.value])) if SubDict[SubjSelect.value][i][0] == RunSelect.value][0]
tp_max                 = [SubDict[SubjSelect.value][i][3] for i in range(0,len(SubDict[SubjSelect.value])) if SubDict[SubjSelect.value][i][0] == RunSelect.value][0]
atlas_name             = 'Craddock_T2Level_0200'
TR                     = 2.0
WL_sec                 = WindowSelect.value
WS_trs                 = 1
WL_trs                 = int(WL_sec / TR)
dim_red_method         = 'PCA'
dim_red_method_percent = 97.5
le_num_dims            = 3
le_k_NN                = 100

# Paths to input and output data
path_ts        = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI','errts.'+SBJ+'.'+atlas_name+'.wl'+str(WL_sec).zfill(3)+'s.fanaticor_ts.1D')
path_outdir    = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI')
out_prefix     = SBJ+'_fanaticor_'+atlas_name+'_wl'+str(WL_sec).zfill(3)+'s_ws'+str(int(WS_trs*TR)).zfill(3)+'s_'+RUN
out_pca_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.pca_obj.pkl')
out_pcats_path = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.pca_ts.pkl')
out_swc_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.swcorr.pkl')
out_lem_path   = osp.join(path_outdir,out_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.le'+str(le_num_dims)+'d_knn'+str(le_k_NN).zfill(3)+'.pkl')

# Prints run information on out script
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

ts_df,Nacq,Nrois,roi_names,ts_xr = load_data(path_ts,RUN,tp_min,tp_max) # Call load data function to load data
print('++ INFO: Time-series loaded into memory [N_acq=%d, N_rois=%d]' % (Nacq, Nrois)) # Show a summary of the data being loaded

# +
# %%time

# Generate Plot of Static Functional connectivity matrix
# ------------------------------------------------------
fc_matrix_plot        = plot_fc_matrix(ts_df,roi_names,'single')
# Generate Timeseries carpet plot
ts_carpet_plot = ts_xr.hvplot.image(cmap='gray', width=1500, colorbar=True, title='ROI Timeseries (carpet plot) - Subject: %s' % SBJ).opts(colorbar_position='bottom')
ts_roi_plot    = ts_df[0].hvplot(cmap='gray',width=1500,height=100)
# Show both plots side-by-side using panel
pn.Row(fc_matrix_plot, pn.Column(ts_carpet_plot,ts_roi_plot))
# -

# ***
# ### 2. Dimensionality Reduction
#
# Here we reduce the dimensionality of the data via PCA. The goal is to have a smaller connectivity matrix, therefore we go from X number of ROIs to a Y number of PCA components, with Y hopefully being much smaller than X.
#
# * How many components are kept depends on the amount of variance we keep (default is 97.5%) 

# +
# %%time

ts_pca_df, pca_plot, pca = reduce_dimensionality_pca(ts_df,dim_red_method_percent,sbj_id=SBJ)
pickle.dump(pca, open(out_pca_path, "wb" ) )
ts_pca_df.to_pickle(out_pcats_path)
# -

pca_plot

# ***
# ### 4. Create SWC Matrix

# +
# %%time

# Create a tukey or tappered window of the appropriate length
# -------------------------------------------------------------
#window = tukey(WL_trs,.2)
window = np.ones((WL_trs,))
pd.DataFrame(window).hvplot(title='Sliding Window Shape',xlabel='Time [TRs]',ylabel='Amplitude')

# +
# %%time

# Compute sliding window correlation
# ----------------------------------
swc_r, swc_Z, winInfo = compute_swc(ts_pca_df,WL_trs,WS_trs,window=window)
xr.DataArray(swc_Z.values.T,dims=['Time [Window ID]','PCA Connection']).hvplot.image(title='SWC Matrix - Fisher Z', cmap='RdBu').redim.range(value=(-1,1)).opts(width=1700)
# -

# ***
# ### 4. Generate Laplacian Embedding

# +
# %%time

# Generate Lapacian embeddings
# ----------------------------
se             = SpectralEmbedding(n_components=le_num_dims, affinity='precomputed', n_jobs=32, random_state=seed)
X_affinity     = kneighbors_graph(swc_Z.T,le_k_NN,include_self=True,n_jobs=32, metric=dis_corr)
X_affinity     = 0.5 * (X_affinity + X_affinity.T)
se_X           = se.fit_transform(X_affinity.toarray())
print ('++ INFO: Embedding Dimensions: %s' % str(se_X.shape))
# -

# Put the embeddings into a dataframe (for saving and plotting)
# -------------------------------------------------------------
LE3D_df = lapacian_dataframe(SubDict,se_X,winInfo,SBJ,RUN,TIME,WL_trs,tp_min,tp_max)
LE3D_df.head()
LE3D_df.to_pickle(out_lem_path)

hv.extension('plotly')
pn.extension('plotly')

player     = pn.widgets.Player(name='Player', start=0, end=winInfo['numWins'], value=1, loop_policy='loop', width=800, step=1)
@pn.depends(player.param.value)
def plot_embed3d(max_win):
    group      = 'Sleep Stage'
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                           LE3D_df['y_norm'][0:max_win],
                           LE3D_df['z_norm'][0:max_win])).opts(color=LE3D_df[group].map(color_key),
                           show_legend=True,
                           size=5,
                           xlim=(-1,1), 
                           ylim=(-1,1), 
                           zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    return output
pn.Column(player,plot_embed3d)

# ### Test with PNAS 2015 Results (for consistency)
# ***

# Load Pre-computed results in MATLAB from one task-based subject from NI2019 
from scipy.io import loadmat
DATAFILE       = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02',
                    'PrcsData_PNAS2015','SBJ06'+'/D02_CTask001/'+'SBJ06'+'_CTask001_WL'+str(30).zfill(3)+'_WS01_NROI0200_dF.mat')
DATAMAT                              = loadmat(DATAFILE)
pnas2015orig_ts_df                   = pd.DataFrame(DATAMAT['origTS'])
pnas2015orig_Nacq,pnas2015orig_Nrois = pnas2015orig_ts_df.shape
pnas2015orig_roi_names              = ['ROI'+str(r+1).zfill(3) for r in range(pnas2015orig_Nrois)]
pnas2015orig_tr                     = DATAMAT['TR'][0][0]
pnas2015orig_ts_xr                  = xr.DataArray(pnas2015orig_ts_df.values,dims=['Time [TRs]','ROIs'])
print('++ Loaded this data: %s' % DATAFILE)

# Generate Plot of Functional connectivity matrix
pnas2015orig_fc_matrix_plot       = plot_fc_matrix(pnas2015orig_ts_df,pnas2015orig_roi_names,'single')
# Generate Timeseries carpet plot
pnas2015orig_ts_carpet_plot         = pnas2015orig_ts_xr.hvplot.image(cmap='gray', width=1500, colorbar=True, title='ROI Timeseries (carpet plot) - Subject: %s' % 'SBJ06').opts(colorbar_position='bottom')
pnas2015orig_ts_roi_plot            = pnas2015orig_ts_df[0].hvplot(cmap='gray',width=1500,height=100)

# +
pn.Row(pnas2015orig_fc_matrix_plot, pnas2015orig_ts_carpet_plot)

# PCA Step
pnas2015python_ts_pca_df, pnas2015python_pca_plot, pnas2015python_pca = reduce_dimensionality_pca(pnas2015orig_ts_df,97.5,sbj_id='SBJ06', n_comp=None)
pnas2015orig_ts_pca_df = pd.DataFrame(DATAMAT['dimRedTS'])
print('++ INFO: PCA (as matlad did it)  --> %d components' % pnas2015orig_ts_pca_df.shape[1])
print('++ INFO: PCA (as python does it) --> %d components' % pnas2015python_ts_pca_df.shape[1])
pnas2015python_pca_plot
# -

pnas2015python_ts_pca_df['PC083'].hvplot(width=1700) * \
pnas2015orig_ts_pca_df[83].hvplot().opts(line_dash='dashed')

# Create a tukey (or tappered window) of the appropriate length
# =============================================================
pnas2015orig_wl_trs   = DATAMAT['WL'][0][0]
pnas2015orig_ws_trs   = DATAMAT['WS'][0][0]
pnas2015python_window = np.ones((pnas2015orig_wl_trs,))
pnas2015orig_swc_Z    = pd.DataFrame(DATAMAT['CB']['snapshots'][0][0].T)
pnas2015python_swc_r, pnas2015python_swc_Z, pnas2015python_winInfo = compute_swc(pnas2015python_ts_pca_df,pnas2015orig_wl_trs,pnas2015orig_ws_trs,window=pnas2015python_window)

xr.DataArray(pnas2015orig_swc_Z.values.T - pnas2015python_swc_Z.values.T,dims=['Time [Window ID]','PCA Connection']).hvplot.image(title='SWC Matrix - Fisher Z', cmap='RdBu_r').redim.range(value=(-1,1)).opts(width=500)

nCom = 3
k_NN = 100
seed = np.random.RandomState(seed=5)

start_time     = time.time()
X = DATAMAT['CB']['snapshots'][0][0]
#X = pnas2015_swc_Z.T
pnas2015_se             = SpectralEmbedding(n_components=nCom, affinity='precomputed', n_jobs=32, random_state=seed)
pnas2015_X_affinity     = kneighbors_graph(X,k_NN,include_self=True,n_jobs=32, metric=dis_corr)
pnas2015_X_affinity     = 0.5 * (pnas2015_X_affinity + pnas2015_X_affinity.T)
pnas2015_se_X           = pnas2015_se.fit_transform(pnas2015_X_affinity.toarray())
end_time                = time.time()
print ('++ INFO: Elapset Time: '+ str(end_time - start_time))
print ('++ INFO: Embedding Dimensions: %s' % str(pnas2015_se_X.shape))

aux_color_int = DATAMAT['winInfo']['color'][0][0]
aux_color_rgb = [ '#%02x%02x%02x' % (int(aux_color_int[i,0]*255), 
                                     int(aux_color_int[i,1]*255), 
                                     int(aux_color_int[i,2]*255)) for i in np.arange(pnas2015python_winInfo['numWins'])]
aux_win_labels = DATAMAT['winInfo']['winNames'][0][0]
embedding_df = pd.DataFrame(columns=['x','y','z','x_norm','y_norm','z_norm','color_int','color_rgb','label'])
embedding_df['x'] = pnas2015_se_X[:,0]
embedding_df['y'] = pnas2015_se_X[:,1]
embedding_df['z'] = pnas2015_se_X[:,2]
embedding_df[['x_norm','y_norm','z_norm']]= embedding_df[['x','y','z']]/embedding_df[['x','y','z']].max()
embedding_df['color_int'] = tuple(aux_color_int)
embedding_df['color_rgb'] = aux_color_rgb
embedding_df['label']     = aux_win_labels
embedding_df.head()
embedding_df.to_pickle('./test_embed.pkl')

hv.extension('plotly')
pn.extension('plotly')

Nwins = embedding_df.shape[0]
player     = pn.widgets.Player(name='Player', start=0, end=Nwins, value=1, loop_policy='loop', width=800, step=1)
@pn.depends(player.param.value)
def plot_embed3d(max_win):
    output = hv.Scatter3D((embedding_df['x_norm'][0:max_win],
                           embedding_df['y_norm'][0:max_win],
                           embedding_df['z_norm'][0:max_win])).opts(color=embedding_df['color_rgb'][0:max_win],
                           size=5, 
                           xlim=(-1,1), 
                           ylim=(-1,1), 
                           zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=800, width=800)
    return output
pn.Column(player,plot_embed3d)

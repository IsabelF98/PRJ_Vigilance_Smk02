# 12/09/2020 - Isabel Fernandez

# # Description: SWC + Embeddings
#
# This notebook performs the following analysis steps:
#
# 1) Load ROI representative time series (those must already exists in text file format)
#
# 2) Dimensionality Reduction from ROI to PCA components (whole time-series)
#
# 3) Compute Sliding Window Correlation based on PCA representative time series
#
# 4) Generate 3D Laplacian Embeddings

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
from utils.emb_func import load_data, lapacian_dataframe
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')
pn.extension()

seed = np.random.RandomState(seed=7) # Seed for embedding
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

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

# Variables for running embeddings
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


# 1. Load ROI Timeseries
# ----------------------
# First, we load the time series for all representative ROIs. If we are wroking with just one run we need to choose just that run from the concatinated data.
ts_df,Nacq,Nrois,roi_names,ts_xr = load_data(path_ts,RUN,tp_min,tp_max) # Call load data function to load data
# Data summery loaded
print('++ INFO: Time-series loaded into memory [N_acq=%d, N_rois=%d]' % (Nacq, Nrois))


# 2. Dimensionality Reduction
# ---------------------------
# Here we reduce the dimensionality of the data via PCA. The goal is to have a smaller connectivity matrix, therefore we go from X number of ROIs to a Y number of PCA components, with Y hopefully being much smaller than X.
# * How many components are kept depends on the amount of variance we keep (default is 97.5%) 
ts_pca_df, pca_plot, pca = reduce_dimensionality_pca(ts_df,dim_red_method_percent,sbj_id=SBJ)
pickle.dump(pca, open(out_pca_path, "wb" ) )
ts_pca_df.to_pickle(out_pcats_path)


# 3. Create SWC Matrix
# --------------------
# Create a tukey or tappered window of the appropriate length
#window = tukey(WL_trs,.2)
window = np.ones((WL_trs,))
# Compute sliding window correlation
swc_r, swc_Z, winInfo = compute_swc(ts_pca_df,WL_trs,WS_trs,window=window)
swc_Z.to_pickle(out_swc_path)


# 4. Generate Laplacian Embedding
# -------------------------------
se             = SpectralEmbedding(n_components=le_num_dims, affinity='precomputed', n_jobs=32, random_state=seed)
X_affinity     = kneighbors_graph(swc_Z.T,le_k_NN,include_self=True,n_jobs=32, metric=dis_corr)
X_affinity     = 0.5 * (X_affinity + X_affinity.T)
se_X           = se.fit_transform(X_affinity.toarray())
print ('++ INFO: Embedding Dimensions: %s' % str(se_X.shape))
# Put the embeddings into a dataframe (for saving and plotting)
LE3D_df = lapacian_dataframe(SubDict,se_X,winInfo,SBJ,RUN,TIME,WL_trs,tp_min,tp_max)
LE3D_df.to_pickle(out_lem_path)
print ('++ INFO: Script Complete')
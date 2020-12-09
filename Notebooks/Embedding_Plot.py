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

# # Embedding Plots
#
# This notebook plots the 3D embedding generated in N01_SWC.ipynb.
# Three widgets are available to select the subject, coresponding run, and window lenth you wish to plot.
# Note that if the data for the subject, run, and window lenght is not availalble the code will alert you and instruct you to run N01_SWC.ipynb for that subject, run, and window lenght. Once the data is available you will be able to plot your embedding.

import pickle
import os
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import panel as pn
from holoviews import dim, opts
hv.extension('plotly')
pn.extension('plotly')

# ***
# ## Create Widgets

# +
# Creates data frame and dictionaly for widgets information 
# ---------------------------------------------------------
# NOTE: Only need to run once

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# Load data frame of valid subjects info
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/valid_run_df.pkl')

# Dictionary of subject with valid runs
SubDict = {}
for i,idx in enumerate(sub_DF.index):
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    if sbj in SubDict.keys():
        SubDict[sbj].append(run)
    else:
        SubDict[sbj] = ['All']
        SubDict[sbj].append(run)

# List of subjects
SubjectList = list(SubDict.keys())

# +
# Widgets for selecting subject run and window legth
# --------------------------------------------------
# NOTE: Only need to run once

SubjSelect = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0]) # Select subject
RunSelect  = pn.widgets.Select(name='Select Run', options=SubDict[SubjSelect.value]) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60]) # Select window lenght

# Updates available runs given SubjSelect value
def update_run(event):
    RunSelect.options = SubDict[event.new]
SubjSelect.param.watch(update_run,'value')

pn.Row(SubjSelect, RunSelect, WindowSelect)
# -

# ***
# ## Load Data

# +
# Subject information
SBJ                    = SubjSelect.value
RUN                    = RunSelect.value
WL_sec                 = WindowSelect.value
atlas_name             = 'Craddock_T2Level_0200'
WS_trs                 = 1
TR                     = 2.0
dim_red_method         = 'PCA'
dim_red_method_percent = 97.5
le_num_dims            = 3
le_k_NN                = 100

#Path to data
PRJDIR       = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
path_datadir = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI')
data_prefix  = SBJ+'_fanaticor_'+atlas_name+'_wl'+str(WL_sec).zfill(3)+'s_ws'+str(int(WS_trs*TR)).zfill(3)+'s_'+RUN
data_path   = osp.join(path_datadir,data_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.le'+str(le_num_dims)+'d_knn'+str(le_k_NN).zfill(3)+'.pkl')
# -

# Try loading data
try:
    LE3D_df = pd.read_pickle(data_path)
except:
    print("++ ERROR: Data for %s, run %s, window lenth %s, DOES NOT EXIST" %(SBJ,RUN,str(WL_sec)))
    print("          Please run N01_SWC.ipynb for this given subject, run, and window lenght")
NTP,arb = LE3D_df.shape

# ***
# ## Plot

player = pn.widgets.Player(name='Player', start=0, end=NTP, value=1, loop_policy='loop', width=800, step=1)
@pn.depends(player.param.value)
def plot_embed3d(max_win):
    output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                           LE3D_df['y_norm'][0:max_win],
                           LE3D_df['z_norm'][0:max_win])).opts(color=LE3D_df['time_color_rgb'][0:max_win],
                           size=5, 
                           xlim=(-1,1), 
                           ylim=(-1,1), 
                           zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    return output
pn.Column(player,plot_embed3d)



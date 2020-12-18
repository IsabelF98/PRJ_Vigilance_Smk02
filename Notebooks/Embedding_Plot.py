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
# Load data frame and create dictionaly for widgets information 
# -------------------------------------------------------------

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# Load data frame of valid subjects info
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl')

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

SubjSelect   = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0]) # Select subject
RunSelect  = pn.widgets.Select(name='Select Run', options=SubDict[SubjSelect.value]) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60]) # Select window lenght
ColorSelect  = pn.widgets.Select(name='Select Color Option', options=['No Color','Time','Sleep','Motion']) # Select color setting for plot

# Updates available runs given SubjSelect value
def update_run(event):
    RunSelect.options = SubDict[event.new]
SubjSelect.param.watch(update_run,'value')


# -

# ***
# ## Load Data

@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def load_data(SBJ,RUN,WL_sec):
    """
    This function loads the data needed for plotting the embeddings.
    The arguments are the subject name, run, and window legth (chosen by the widgets).
    The fuction returns a pandas data frame of the data.
    """
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
    LE3D_df = pd.read_pickle(data_path)
    return LE3D_df


@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def get_num_tp(SBJ,RUN,WL_sec):
    """
    This function outputs the number of time points in the selected data.
    To load the data load_data() fuction is called.
    The arguments are the subject name, run, and window legth (chosen by the widgets).
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec)
    value,arb = LE3D_df.shape
    return value


# ***
# ## Plot

# +
# Time player for embedding plot
player = pn.widgets.Player(name='Player', start=0, end=get_num_tp(SubjSelect.value,RunSelect.value,WindowSelect.value), value=1,
                           loop_policy='loop', width=800, step=1)

#def update_player(event1,event2,event3):
#    player.end = get_num_tp(event1.new,event2.new,event3.new)
#SubjSelect.param.watch(update_player,'value')
#RunSelect.param.watch(update_player,'value')
#WindowSelect.param.watch(update_player,'value')


@pn.depends(player.param.value,SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value,ColorSelect.param.value)
def plot_embed3d(max_win,SBJ,RUN,WL_sec,COLOR):
    """
    This function plots the embeddings on a 3D plot.
    To load the data load_data() fuction is called.
    The dimensions of the plot "max_win" (time) is determined by the player.
    The subject, run, window lenght, and color scheme of the plot is dermined by widgets selected.
    The output of the function is the plot generated by holoviews with extension plotly.
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec)
    if COLOR == 'No Color':
        output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                               LE3D_df['y_norm'][0:max_win],
                               LE3D_df['z_norm'][0:max_win])).opts(
                               size=5,
                               xlim=(-1,1), 
                               ylim=(-1,1), 
                               zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    if COLOR == 'Time':
        output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                               LE3D_df['y_norm'][0:max_win],
                               LE3D_df['z_norm'][0:max_win])).opts(color=LE3D_df.index,
                               colorbar=True,
                               size=5,
                               xlim=(-1,1), 
                               ylim=(-1,1), 
                               zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    if COLOR == 'Sleep':
        color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                               LE3D_df['y_norm'][0:max_win],
                               LE3D_df['z_norm'][0:max_win])).opts(color=LE3D_df['Sleep Stage'].map(color_key),
                               show_legend=True,
                               size=5,
                               xlim=(-1,1), 
                               ylim=(-1,1), 
                               zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    if COLOR == 'Motion':
        output = hv.Scatter3D((LE3D_df['x_norm'][0:max_win],
                               LE3D_df['y_norm'][0:max_win],
                               LE3D_df['z_norm'][0:max_win])).opts(color=LE3D_df['Motion'],
                               colorbar=True,
                               size=5,
                               xlim=(-1,1), 
                               ylim=(-1,1), 
                               zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=600, width=600)
    return output


# -

# ***
# ## Plot Display

pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),player,plot_embed3d)

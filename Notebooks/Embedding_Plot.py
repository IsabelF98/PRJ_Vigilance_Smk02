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
RunSelect    = pn.widgets.Select(name='Select Run', options=SubDict[SubjSelect.value]) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60]) # Select window lenght
ColorSelect  = pn.widgets.Select(name='Select Color Option', options=['No Color','Time/Run','Sleep','Motion']) # Select color setting for plot

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
    plot_df = LE3D_df[['x_norm','y_norm','z_norm']].copy()
    if COLOR == 'No Color':
        df = LE3D_df[['x_norm','y_norm','z_norm']].copy()
        output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm'])
        output = output.opts(size=5,
                             xlim=(-1,1), 
                             ylim=(-1,1), 
                             zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700)
    if COLOR == 'Time/Run':
        if RUN != 'All':
            df = LE3D_df[['x_norm','y_norm','z_norm']].copy()
            df['Time [sec]'] = 2*LE3D_df.index
            df = df.infer_objects()
            output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm'],vdims='Time [sec]')
            output = output.opts(color='Time [sec]',
                                 cmap='plasma',
                                 colorbar=True,
                                 clim=(0,get_num_tp(SBJ,RUN,WL_sec)),
                                 size=5,
                                 xlim=(-1,1), 
                                 ylim=(-1,1), 
                                 zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700)
        else:
            df = LE3D_df[['x_norm','y_norm','z_norm','Run']].copy()
            df = df.infer_objects()
            color_key  = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple','WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
            for i,idx in enumerate(df.index):
                df.loc[idx,'color'] = color_key[df.loc[idx,'Run']]
            dict_plot={t:hv.Scatter3D(df[0:max_win].query(f'Run=="{t}"'),kdims=['x_norm','y_norm','z_norm'],vdims=['Run','color']).opts(show_legend=True,color='color',size=5,fontsize={'legend':8}) for t in df[0:max_win].Run.unique()}
            output = hv.NdOverlay(dict_plot).opts(
                              xlim=(-1,1), 
                              ylim=(-1,1), 
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=700, width=700)
            output.get_dimension('Element').label='Run '
    if COLOR == 'Sleep':
        df = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy()
        df = df.infer_objects()
        df = df.rename(columns={"Sleep Stage": "Sleep_Stage"})
        color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        for i,idx in enumerate(df.index):
                df.loc[idx,'color'] = color_key[df.loc[idx,'Sleep_Stage']]
        dict_plot={t:hv.Scatter3D(df[0:max_win].query(f'Sleep_Stage=="{t}"'),kdims=['x_norm','y_norm','z_norm'],vdims=['Sleep_Stage','color']).opts(show_legend=True,color='color',size=5,fontsize={'legend':8}) for t in df[0:max_win].Sleep_Stage.unique()}
        output = hv.NdOverlay(dict_plot).opts(
                              xlim=(-1,1), 
                              ylim=(-1,1), 
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=700, width=700)
        output.get_dimension('Element').label='Sleep_Stage '
    if COLOR == 'Motion':
        df = LE3D_df[['x_norm','y_norm','z_norm','Motion']].copy()
        max_FD = df['Motion'].max()
        df = df.infer_objects()
        output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm'],vdims='Motion')
        output = output.opts(opts.Scatter3D(color='Motion',
                                            cmap='jet',
                                            colorbar=True,
                                            clim=(0,max_FD),
                                            size=5,
                                            xlim=(-1,1), 
                                            ylim=(-1,1), 
                                            zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700))
    return output


# -

# ***
# ## Plot Display

pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),player,plot_embed3d)

# ***
# ## Testing

data = np.random.randn(10, 3).cumsum(axis=0)
df   = pd.DataFrame(data,columns=['x','y','z'])
df['type'] = ['A','A','B','C','A','C','C','B','C','A']
color_key  = {'A':'red', 'B':'blue','C':'green'}
for i,idx in enumerate(df.index):
    df.loc[idx,'color'] = color_key[df.loc[idx,'type']]
dict_plot={t:hv.Scatter3D(df.query(f'type=="{t}"'),kdims=['x','y','z'],vdims=['type','color']).opts(show_legend=True,color='color',size=3) for t in df.type.unique()}
h=hv.NdOverlay(dict_plot)
h.get_dimension('Element').label='type '
h
# +
data_dict={'sub-01':[('run1',100),('run2',100),('run3',60)],'sub-02':[('run1',200),('run2',100),('run3',50)],'sub-03':[('run1',100),('run2',400)]}
subject_list = list(data_dict.keys())

subject_select = pn.widgets.Select(name='Select Subject',options=subject_list,value=subject_list[0])
run_select     = pn.widgets.Select(name='Select Run', options=[data_dict[subject_select.value][i][0] for i in range(0,len(data_dict[subject_select.value]))])
def update_run(event):
    run_select.options = [data_dict[event.new][i][0] for i in range(0,len(data_dict[event.new]))]
subject_select.param.watch(update_run,'value')

@pn.depends(subject_select.param.value,run_select.param.value)
def get_num_tp(SBJ,RUN):
    num_tp = [data_dict[subject_select.value][i][1] for i in range(0,len(data_dict[subject_select.value])) if data_dict[subject_select.value][i][0] == run_select.value][0]
    return num_tp

player = pn.widgets.Player(name='Player', start=0, end=get_num_tp(subject_select.value,run_select.value), value=1,loop_policy='loop', width=800, step=1)
def update_player(event1,event2):
    player.end = get_num_tp(event1.new,event2.new)
subject_select.param.watch(update_player,'value')
run_select.param.watch(update_player,'value')

@pn.depends(player.param.value)
def print_player_value(value):
    value = str(value)
    markdown = pn.pane.Markdown(value)
    return markdown

pn.Column(pn.Row(subject_select,run_select),player,print_player_value)

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
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
from holoviews.plotting.links import DataLink
import plotly.express as px
import plotly.graph_objects as go
import panel as pn
import scipy
from scipy.spatial.distance import pdist, squareform
from holoviews import dim, opts
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')
pn.extension('plotly')

# Call port tunnel for gui display
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# ***
# ## Create Widgets

# +
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

# Load data frame of valid subjects info
SubDict = {} # Empty dictionary
for i,idx in enumerate(sub_DF.index): # Iterate through each row of data frame
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    time = sub_DF.loc[idx]['Time']
    if sbj in SubDict.keys():
        SubDict[sbj].append((run,time)) # Add run tuple (described above)
    else:
        SubDict[sbj] = [(run,time)] # If subject is not already in the directory a new element is created
        
SubjectList = list(SubDict.keys()) # list of subjects   

# Add 'All' option to subject diction for each subject. 'All' meaning the concatinated data
for sbj in SubjectList:
    SubDict[sbj].append(('All',sum(SubDict[sbj][i][1] for i in range(0,len(SubDict[sbj])))))

# +
# Widgets for selecting subject run and window legth
SubjSelect   = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0],width=200) # Select subject
RunSelect    = pn.widgets.Select(name='Select Run', options=[SubDict[SubjSelect.value][i][0] for i in range(0,len(SubDict[SubjSelect.value]))],width=200) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60],width=200) # Select window lenght
ColorSelect  = pn.widgets.Select(name='Select Color Option', options=['No Color','Time/Run','Sleep','Motion'],width=200) # Select color setting for plot

# Updates available runs given SubjSelect value
def update_run(event):
    RunSelect.options = [SubDict[event.new][i][0] for i in range(0,len(SubDict[event.new]))]
SubjSelect.param.watch(update_run,'value')


# -

# ***
# ## Load Data

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
# ## 3D Plot

# +
# Time player for embedding plot
player = pn.widgets.Player(name='Player', start=0, end=get_num_tp(SubjSelect.value,RunSelect.value,WindowSelect.value), value=1,
                           loop_policy='loop', width=800, step=1)

# Update player end value for new run
@pn.depends(SubjSelect, RunSelect, WindowSelect, watch=True)
def update_player(SBJ,RUN,WL_sec):
    end_value = get_num_tp(SBJ,RUN,WL_sec) # Get number of time points from get_num_tp() function
    player.value = min(player.value, end_value) # Update player value to last player value or new end value
    player.end = end_value # Update end value


# -

# Make function dependint on player, subject, run, and window length values
@pn.depends(player.param.value,SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value,ColorSelect.param.value)
def plot_embed3d(max_win,SBJ,RUN,WL_sec,COLOR):
    """
    This function plots the embeddings on a 3D plot.
    To load the data load_data() fuction is called.
    The dimensions of the plot "max_win" is determined by the player so we can watch the plot progress over time.
    The subject, run, window lenght, and color scheme of the plot is determined by widgets selected.
    The output of the function is the plot generated by plotly.
    """
    
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load data to be plotted
    LE3D_df = LE3D_df.infer_objects() # Infer objects to be int, float, or string apropriatly
    
    title = 'Laplacian Embedings for '+SBJ+' '+RUN+' and WL '+str(WL_sec)+' [sec]' # Plot title
    
    # Plots 3D scatter with no color
    if COLOR == 'No Color':
        color = None
        color_continuous_scale = None
        color_discrete_map = None
        range_color = None
        
    # Plots 3D peramiters for color based on time (if a single run) or based on run (if all runs)
    if COLOR == 'Time/Run':
        if RUN != 'All': # A single run
            color = range(0,max_win)
            color_continuous_scale = 'viridis'
            color_discrete_map = None
            range_color = [0,LE3D_df.shape[0]]
        else: # All the runs concatinated
            # color_map defines the colors coresponding to each run
            color = 'Run'
            color_discrete_map  = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple',
                                      'WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
            color_continuous_scale = None
            range_color = None
    
    # Plot 3D peramiters with coloring by sleep stage
    if COLOR == 'Sleep':
        # color_map defines the colors coresponding to each sleep stage
        color = 'Sleep Stage'
        color_discrete_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        color_continuous_scale = False
        range_color = None
   
    # Plot 3D peramiters with coloring by motion (average framewise displacment over each window)
    if COLOR == 'Motion':
        color = 'Motion'
        color_continuous_scale = 'jet'
        color_discrete_map = None
        range_color = [0,LE3D_df['Motion'].max()]
        
    output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color=color,color_continuous_scale=color_continuous_scale,
                           color_discrete_map=color_discrete_map,range_color=range_color,range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],
                           width=700,height=600,opacity=0.9,title=title)
    
    output = output.update_traces(marker=dict(size=5,line=dict(width=0)))
    
    return output


# + jupyter={"source_hidden": true}
# Make function dependint on player, subject, run, and window length values
@pn.depends(player.param.value,SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value,ColorSelect.param.value)
def plot_embed3d(max_win,SBJ,RUN,WL_sec,COLOR):
    """
    This function plots the embeddings on a 3D plot.
    To load the data load_data() fuction is called.
    The dimensions of the plot "max_win" is determined by the player so we can watch the plot progress over time.
    The subject, run, window lenght, and color scheme of the plot is determined by widgets selected.
    The output of the function is the plot generated by plotly.
    """
    
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load data to be plotted
    LE3D_df = LE3D_df.infer_objects() # Infer objects to be int, float, or string apropriatly
    
    title = 'Laplacian Embedings for '+SBJ+' '+RUN+' and WL '+str(WL_sec)+' [sec]' # Plot title
    
    # Plots 3D scatter with no color
    if COLOR == 'No Color':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],
                               width=700,height=600,opacity=0.7,title=title)
    
    # Plots 3D scatter with color based on time (if a single run) or based on run (if all runs)
    if COLOR == 'Time/Run':
        if RUN != 'All': # A single run
            output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color=range(0,max_win), range_color=[0,LE3D_df.shape[0]],
                                   range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7,title=title)
        else: # All the runs concatinated
            # color_map defines the colors coresponding to each run
            color_map = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple',
                                      'WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
            output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Run',color_discrete_map=color_map,
                                   range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7,title=title)
    
    # Plot 3D with coloring by sleep stage
    if COLOR == 'Sleep':
        # color_map defines the colors coresponding to each sleep stage
        color_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Sleep Stage',color_discrete_map=color_map,
                               range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7,title=title)
   
    # Plot 3D with coloring by motion (average framewise displacment over each window)
    if COLOR == 'Motion':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Motion',color_continuous_scale='jet',
                               range_color=[0,LE3D_df['Motion'].max()],range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,
                               height=600,opacity=0.7,title=title)
        
    output = output.update_traces(marker=dict(size=5,line=dict(width=0)))
    
    return output


# -

# ***
# ## Euclidean Distance Matrix

# Make function dependint on subject, run, and window length values
@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def distance_matrix(SBJ,RUN,WL_sec):
    """
    This fuction plots a heat map of the distnaces of each window for a given run.
    The inputs for the fuction (subject, run, and window leght) allows the user to choose what run and window leghth
    they with to plot for a given subject.
    The distance between two windows (i.e. points on the 3D plot) is computed using numpys squareform(pdist()).
    The fuction plots the heat map using holoviews hv.Image().
    The x and y axes of the plot are the two windows in which you are finding the distance.
    The z value is that distance.
    A plot of the sleep staging segments are ploted along the x and y axis of the image using hv.Segments().
    If all runs are being displayed a plot of the run segments are ploted along the x and y axis of the image using hv.Segments().
    """
    LE3D_df    = load_data(SBJ,RUN,WL_sec) # Load embedding data
    data_path  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl')
    sleep_segments_df = pd.read_pickle(data_path) 
    data_df    = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy() # New data frame of only x_norm, y_norm, and z_norm values
    
    data_array = data_df[['x_norm','y_norm','z_norm']].to_numpy() # Data as a numpy array
    dist_array = squareform(pdist(data_array, 'euclidean')) # Calculate distance matrix and rehape into one vecotr
    dist_array = xr.DataArray(dist_array,dims=['Time [Window ID]','Time [Window ID] Y']) # Distances as x_array data frame
    
    sleep_color_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'} # Color key for sleep staging
    
    # Plot of sleep staging segements along the x and y axis
    sleep_seg_x = hv.Segments(sleep_segments_df, [hv.Dimension('start',range=(-5,data_df.shape[0])), hv.Dimension('start_event',range=(-5,data_df.shape[0])),
                              'end', 'end_event'],'stage').opts(color='stage',cmap=sleep_color_map,line_width=8,show_legend=False)
    sleep_seg_y = hv.Segments(sleep_segments_df, [hv.Dimension('start_event',range=(-5,data_df.shape[0])), hv.Dimension('start',range=(-5,data_df.shape[0])),
                              'end_event', 'end'],'stage').opts(color='stage',cmap=sleep_color_map,line_width=8,show_legend=False)
    
    # If plotting all runs add segent to x and y axis for coloring by run
    if RUN == 'All':
        run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ])-1)] # List of all runs
        time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ])-1)] # List of all run lenghts in TR's (in the same order as runs in list above)

        WL_trs = int(WL_sec/2) # Window length in TR's

        run_segments_df = pd.DataFrame(columns=['run','start','end']) # Emptly data frame for segment legths of runs
        
        # For each run a row is appended into the data frame created above with the run name and the start and end window of the data
        x=0 # Starting at 0th TR
        for i in range(len(run_list)):
            time = time_list[i] # Number of TR's in run
            run  = run_list[i] # Name of run
            end=x+time-WL_trs # End TR of run
            if i == len(run_list)-1: # If its the last run no need to append inbetween run
                run_segments_df = run_segments_df.append({'run':run,'start':x,'end':end}, ignore_index=True)
            else: 
                run_segments_df = run_segments_df.append({'run':run,'start':x,'end':end}, ignore_index=True) # Append run info
                x=end+1
                run_segments_df = run_segments_df.append({'run':'Inbetween Runs','start':x,'end':(x-1)+(WL_trs-1)}, ignore_index=True) # Append inbetween run info
                x=x+(WL_trs-1)

        # Add 0.5 to each end of segment to span entire heat map
        run_segments_df['start'] = run_segments_df['start'] - 0.5 
        run_segments_df['end']   = run_segments_df['end'] + 0.5
        
        # 'start_event' and 'end_event' represent the axis along which the segments will be (-30 so it is not on top of the heat map)
        run_segments_df['start_event'] = -30
        run_segments_df['end_event']   = -30
        
        # Color key for runs
        run_color_map = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple','WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
        
        # Plot of run segements along the x and y axis
        run_seg_x = hv.Segments(run_segments_df, [hv.Dimension('start',range=(-40,data_df.shape[0])), hv.Dimension('start_event',range=(-40,data_df.shape[0])),
                                'end', 'end_event'],'run').opts(color='run',cmap=run_color_map,line_width=8,show_legend=False)
        run_seg_y = hv.Segments(run_segments_df, [hv.Dimension('start_event',range=(-20,data_df.shape[0])), hv.Dimension('start',range=(-20,data_df.shape[0])),
                                'end_event', 'end'],'run').opts(color='run',cmap=run_color_map,line_width=8,show_legend=False)
        
        segment_plot = (sleep_seg_x*sleep_seg_y*run_seg_x*run_seg_y).opts(xlabel=' ',ylabel=' ',show_legend=False) # All segments concatinated (including runs)
    else:
        segment_plot = (sleep_seg_x*sleep_seg_y).opts(xlabel=' ',ylabel=' ',show_legend=False) # All segments concatinated (not including runs)
    
    # Plot heat map using hv.HeatMap() with hover tool
    plot = rasterize(hv.Image(dist_array,bounds=(-0.5,-0.5,data_df.shape[0]-0.5,data_df.shape[0]-0.5)).opts(cmap='jet',ylabel='Time [Window ID]'))
    
    # Overlay segment plots and heat map
    output = (plot*segment_plot).opts(width=500,height=400)
    return output


# ***
# ## Motion Trace

# Make function dependint on subject, run, and window length values
@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def motion_trace(SBJ,RUN,WL_sec):
    """
    This fuction plots the motion trace for a given subject run and window length.
    The fuction plots the average framewise displacemnt over a window using holovies hv.Curve() function.
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load embedding data
    #Plot the Framewise Displacment over windows
    output = hv.Curve(LE3D_df['Motion']).opts(xlabel='Time [Window ID]',ylabel='Framewise Displacement',width=500,height=200)
    return output


# ***
# ## Plot Display

# Add a text box to describe run
@pn.depends(RunSelect.param.value)
def run_description(RUN):
    """
    This function displays a markdown panel with a discription of the run being displayed.
    """
    if RUN == 'All':
        output = pn.pane.Markdown("""
                                  ### All:
                                  This is the concatenated  data of all the runs. The order in which they are concatenated, 
                                  if such runs exist for that subject, are SleepAscending, Sleep Descending, Sleep RSER, 
                                  Wake Ascending, Wake Descending, and Wake RSER.
                                  """, width=500)
    if RUN == 'SleepAscending':
        output = pn.pane.Markdown("""
                                  ### Sleep Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """, width=500)
    if RUN == 'SleepDescending':
        output = pn.pane.Markdown("""
                                  ### Sleep Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """, width=500)
    if RUN == 'SleepRSER':
        output = pn.pane.Markdown("""
                                  ### Sleep RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to fall 
                                  asleep with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones. 
                                  """, width=500)
    if RUN == 'WakeAscending':
        output = pn.pane.Markdown("""
                                  ### Wake Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """, width=500)
    if RUN == 'WakeDescending':
        output = pn.pane.Markdown("""
                                  ### Wake Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """, width=500)
    if RUN == 'WakeRSER':
        output = pn.pane.Markdown("""
                                  ### Wake RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to stay 
                                  awake with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones.
                                  """, width=500)
    return output


# Make function dependint on subject, run, and window length values
@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def dist_mot_trace(SBJ,RUN,WL_sec):
    dist = distance_matrix(SBJ,RUN,WL_sec)
    mot  = motion_trace(SBJ,RUN,WL_sec)
    output = (dist + mot).cols(1)
    return output


# Display widgets player and plots
dash = pn.Column(pn.Row(pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),player),run_description),
          pn.Row(plot_embed3d,dist_mot_trace))

# Display widgets player and plots
dash = pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect)

# Creat http for gui
dash_server = dash.show(port=port_tunnel, open=False)

# Stop gui
dash_server.stop()

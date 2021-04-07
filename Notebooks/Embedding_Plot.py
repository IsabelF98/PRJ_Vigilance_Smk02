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
import plotly.express as px
import panel as pn
import scipy
from scipy.spatial.distance import pdist, squareform
from holoviews import dim, opts
from holoviews.operation.datashader import rasterize
from holoviews.plotting.links import DataLink
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
RunSelect    = pn.widgets.Select(name='Select Run', options=[SubDict[SubjSelect.value][i][0] for i in range(0,len(SubDict[SubjSelect.value]))],value=[SubDict[SubjSelect.value][i][0] for i in range(0,len(SubDict[SubjSelect.value]))][0],width=200) # Select run for chosen subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60],width=200) # Select window lenght
ColorSelect  = pn.widgets.Select(name='Select Color Option', options=['No Color','Time/Run','Sleep','Motion'],width=200) # Select color setting for plot

# Updates available runs given SubjSelect value
@pn.depends(SubjSelect.param.value, watch=True)
def update_run(SBJ):
    pre_select_val = RunSelect.value # Last run value
    runs = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ]))] # List of runs for that subject
    if pre_select_val not in runs: # If last run is NOT a valid run for that subject
        RunSelect.options = runs # New options for runs
        RunSelect.value = runs[0] # First value of run list
    else: # If run is a valid run for that subject
        RunSelect.options = runs # New options for runs
        RunSelect.value = pre_select_val # Same run as last run value


# -

# ***
# ## Load Data and Get Data Info

# Load sleep order data at a pandas data frame
data_path  = osp.join(PRJDIR,'PrcsData','all','run_order.csv')
run_order_df = pd.read_csv(data_path)[['subject','run']].copy()


def load_data(SBJ,RUN,WL_sec):
    """
    This function loads the data needed for plotting the embeddings.
    The arguments are the subject name, run, and window legth (chosen by the widgets).
    The fuction returns a pandas data frame of the data.
    """
    # Data Info
    atlas_name             = 'Craddock_T2Level_0200'
    WS_trs                 = 1
    TR                     = 2.0
    dim_red_method         = 'PCA'
    dim_red_method_percent = 97.5
    le_num_dims            = 3
    le_k_NN                = 100

    #Path to data
    path_datadir = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI')
    data_prefix  = SBJ+'_fanaticor_'+atlas_name+'_wl'+str(WL_sec).zfill(3)+'s_ws'+str(int(WS_trs*TR)).zfill(3)+'s_'+RUN
    data_path    = osp.join(path_datadir,data_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.le'+str(le_num_dims)+'d_knn'+str(le_k_NN).zfill(3)+'.pkl')
    LE3D_df      = pd.read_pickle(data_path)
    return LE3D_df


def load_sleep_stage_data(SBJ,RUN,WL_sec):
    """
    This function loads the sleep staging data on a TR to TR basis for each subject and run as a pandas data frame
    Given the window length of the data selected the data frame with be displaced so the center TR of the TR to the
    right of the center of each window is allighned
    The data frame has two columns window cneter and sleep stage
    """
    WL_trs = WL_sec/2 # Window lenght in terms of TR (TR = 2sec)
    
    if (WL_trs % 2) == 0: # If window length is even set window to center TR
        disp = int(WL_trs/2) # How much to displace the data by (i.e 1/2 a window)
    else: # If window length is even set window to the TR right of center
        disp = int((WL_trs+1)/2) # How much to displace the data by (i.e 1/2 a window)
    
    if RUN != 'All': # If a single run just load single run data
        DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        temp_df  = pd.read_pickle(DATADIR)
        
    else: # If all runs concatinate all runs data
        run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ]))]
        run_list.remove('All')
        temp_df  = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds','stage'])
        # Append each runs sleep stage data to end of temp_df
        for r in run_list:
            DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+r+'_EEG_sleep.pkl')
            run_sleep_df = pd.read_pickle(DATADIR)
            temp_df = temp_df.append(run_sleep_df).reset_index(drop = True)
    
    TIME = int(temp_df.shape[0]) # How many TR's in data
    
    # Create empty data frame for sleep staging data with displaced index
    sleep_df = pd.DataFrame(columns=['Time [Window ID]','Sleep Stage'],index=range(-disp,TIME-disp))
    
    sleep_df['Time [Window ID]'] = sleep_df.index # Column of window centers (same as index)
    sleep_list = list(temp_df['stage']) # Set all sleep stages as list
    sleep_df['Sleep Stage'] = sleep_list # Append list for sleep stages of data frame
    
    return sleep_df


def get_num_tp(SBJ,RUN,WL_sec):
    """
    This function outputs the number of windows in the selected data.
    To load the data load_data() fuction is called.
    The arguments are the subject name, run, and window legth (chosen by the widgets).
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec)
    value = LE3D_df.shape[0]
    return value


# ***
# ## 3D Plot

# +
# Time player for embedding plot
player = pn.widgets.Player(name='Player', start=0, end=get_num_tp(SubjSelect.value,RunSelect.value,WindowSelect.value),
                           value=get_num_tp(SubjSelect.value,RunSelect.value,WindowSelect.value), loop_policy='loop', width=800, step=1)

# Update player end value for new run
@pn.depends(SubjSelect, RunSelect, WindowSelect, watch=True)
def update_player(SBJ,RUN,WL_sec):
    end_value = get_num_tp(SBJ,RUN,WL_sec) # Get number of time points from get_num_tp() function
    player.value = end_value # Update player value to last player value or new end value
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
        
    # Plots 3D scatter with no color
    if COLOR == 'No Color':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],
                               width=700,height=600,opacity=0.7)
    
    # Plots 3D scatter with color based on time (if a single run) or based on run (if all runs)
    if COLOR == 'Time/Run':
        if RUN != 'All': # A single run
            output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color=range(0,max_win), range_color=[0,LE3D_df.shape[0]],
                                   range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7)
        else: # All the runs concatinated
            # color_map defines the colors coresponding to each run
            color_map = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
                                      'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}
            output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Run',color_discrete_map=color_map,
                                   range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7)
    
    # Plot 3D with coloring by sleep stage
    if COLOR == 'Sleep':
        # color_map defines the colors coresponding to each sleep stage
        color_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Sleep Stage',color_discrete_map=color_map,
                               range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,height=600,opacity=0.7)
   
    # Plot 3D with coloring by motion (average framewise displacment over each window)
    if COLOR == 'Motion':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='mean FD',color_continuous_scale='jet',
                               range_color=[0,LE3D_df['mean FD'].max()],range_x=[-1,1],range_y=[-1,1],range_z=[-1,1],width=700,
                               height=600,opacity=0.8)
    
    output = output.update_traces(marker=dict(size=5,line=dict(width=0))) # No outline on points

    output = output.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),uirevision=True) # Change legend position of data to top horizontal
    
    return pn.pane.Plotly(output) # Return plot in panel plotly format (solves color issue!)


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
    data_path  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl') # path to segment data
    sleep_segments_df = pd.read_pickle(data_path) # Load segment data
    data_df    = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy()# New data frame of only x_norm, y_norm, and z_norm values
    num_win    = data_df.shape[0] # Number of windwos in data
    
    data_array = data_df[['x_norm','y_norm','z_norm']].to_numpy() # Data as a numpy array
    dist_array = squareform(pdist(data_array, 'euclidean')) # Calculate distance matrix and rehape into one vecotr
    dist_array = xr.DataArray(dist_array,dims=['Time [Window ID]','Time [Window ID] Y']) # Distances as x_array data frame
    
    sleep_color_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'} # Color key for sleep staging
    
    # Plot of sleep staging segements along the x and y axis
    # Range is from (-10, num_win) so we have space to display segments
    sleep_seg_x = hv.Segments(sleep_segments_df, [hv.Dimension('start',range=(-10,num_win)), hv.Dimension('start_event',range=(-5,num_win)),
                              'end', 'end_event'],'stage').opts(color='stage',cmap=sleep_color_map,line_width=10,show_legend=False)
    sleep_seg_y = hv.Segments(sleep_segments_df, [hv.Dimension('start_event',range=(-10,num_win)), hv.Dimension('start',range=(-5,num_win)),
                              'end_event', 'end'],'stage').opts(color='stage',cmap=sleep_color_map,line_width=10,show_legend=False)
    
    # If plotting all runs add segent to x and y axis for coloring by run
    if RUN == 'All':
        run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ])-1)] # List of all runs
        time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ])-1)] # List of all run lenghts in TR's (in the same order as runs in list above)

        WL_trs = int(WL_sec/2) # Window length in TR's (TR = 2.0 sec)

        run_segments_df = pd.DataFrame(columns=['run','start','end']) # Emptly data frame for segment legths of runs
        
        # For each run a row is appended into the data frame created above (run_segments_df) with the run name and the start and end window of the run
        # For the windows that overlap runs the run will be called 'Inbetween Runs'
        x=0 # Starting at 0th window
        for i in range(len(run_list)):
            time = time_list[i] # Number of windows in run
            run  = run_list[i] # Name of run
            end=x+time-WL_trs # Last window of run
            if i == len(run_list)-1: # If its the last run no need to append inbetween run
                run_segments_df = run_segments_df.append({'run':run,'start':x,'end':end}, ignore_index=True) # Append run info
            else: 
                run_segments_df = run_segments_df.append({'run':run,'start':x,'end':end}, ignore_index=True) # Append run info
                x=end+1
                run_segments_df = run_segments_df.append({'run':'Inbetween Runs','start':x,'end':(x-1)+(WL_trs-1)}, ignore_index=True) # Append inbetween run info
                x=x+(WL_trs-1)

        # Add 0.5 to each end of segment to span entire heat map
        run_segments_df['start'] = run_segments_df['start'] - 0.5 
        run_segments_df['end']   = run_segments_df['end'] + 0.5
        
        # 'start_event' and 'end_event' represent the axis along which the segments will be (-50 so it is not on top of the heat map or sleep segments)
        run_segments_df['start_event'] = -50
        run_segments_df['end_event']   = -50
        
        # Color key for runs
        run_color_map = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
                         'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}
        
        # Plot of run segements along the x and y axis
        # Range is from (-80, num_win) so we have space to display both segments
        run_seg_x = hv.Segments(run_segments_df, [hv.Dimension('start',range=(-80,num_win)), hv.Dimension('start_event',range=(-80,num_win)),
                                'end', 'end_event'],'run').opts(color='run',cmap=run_color_map,line_width=10,show_legend=False)
        run_seg_y = hv.Segments(run_segments_df, [hv.Dimension('start_event',range=(-80,num_win)), hv.Dimension('start',range=(-80,num_win)),
                                'end_event', 'end'],'run').opts(color='run',cmap=run_color_map,line_width=10,show_legend=False)
        
        segment_plot = (sleep_seg_x*sleep_seg_y*run_seg_x*run_seg_y).opts(xlabel=' ',ylabel=' ',show_legend=False) # All segments (run and sleep) overlayed
    else:
        segment_plot = (sleep_seg_x*sleep_seg_y).opts(xlabel=' ',ylabel=' ',show_legend=False) # All segments (not including runs)overlayed
    
    # Plot heat map using hv.Image
    # Set bounds to (-0.5,-0.5,num_win-0.5,num_win-0.5) to corespond with acurate windows
    plot = hv.Image(dist_array,bounds=(-0.5,-0.5,num_win-0.5,num_win-0.5)).opts(cmap='jet',colorbar=True,ylabel='Time [Window ID]')
    
    # Overlay segment plots and heat map
    output = (plot*segment_plot).opts(width=600,height=390)

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
    LE3D_df['Time [Window ID]'] = LE3D_df.index
    
    # Depending on segments for distance matrix add buffer space to the left of motion so x axis of matrix and x axis of motion trace alighn
    if RUN == 'All':
        xlim=(-80,LE3D_df.shape[0])
    else:
        xlim=(-10,LE3D_df.shape[0])
    
    #Plot the Framewise Displacment over windows
    output = hv.Curve(LE3D_df,vdims=['mean FD'],kdims=['Time [Window ID]']).opts(xlabel='Time [Window ID]',ylabel='Framewise Disp.',
                                                                                 width=600,height=150,xlim=xlim,ylim=(0,1))
    return output


# ***
# ## Sleep Staging Trace by TR

@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def sleep_stage_trace(SBJ,RUN,WL_sec):
    """
    This function plots the sleep staging of the data as a line plot with windows on the x axis and sleep stage on the y axis
    """
    sleep_df = load_sleep_stage_data(SBJ,RUN,WL_sec)
    output   = hv.Curve(sleep_df,kdims=['Time [Window ID]'],vdims=['Sleep Stage']).opts(width=600,height=150)
    return output


# ***
# ## Plot Display

@pn.depends(SubjSelect.param.value)
def run_order(SBJ):  
    sbj1 = SBJ[5:7]

    run_list = []
    for i,idx in enumerate(run_order_df.index):
        sbj2 = run_order_df.loc[idx]['subject'][3:5]
        run  = run_order_df.loc[idx]['run']
        if sbj1 == sbj2:
            run_list.append(run)
    
    run_str = "####Run Order:"
    
    for run in run_list:
        if run == run_list[-1]:
            run_str = run_str+' '+run
        else:
            run_str = run_str+' '+run+','
    
    output = pn.pane.Markdown(run_str,width=500)
    return output


# Add a text box to describe run
@pn.depends(RunSelect.param.value)
def run_description(RUN):
    """
    This function displays a markdown panel with a discription of the run being displayed.
    """
    
    width = 500 # Markdown panel width
    if RUN == 'All':
        output = pn.pane.Markdown("""
                                  ### All:
                                  This is the concatenated  data of all the runs. The order in which they are concatenated, 
                                  if such runs exist for that subject, are SleepAscending, Sleep Descending, Sleep RSER, 
                                  Wake Ascending, Wake Descending, and Wake RSER.
                                  """, width=width)
    if RUN == 'SleepAscending':
        output = pn.pane.Markdown("""
                                  ### Sleep Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """, width=width)
    if RUN == 'SleepDescending':
        output = pn.pane.Markdown("""
                                  ### Sleep Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """, width=width)
    if RUN == 'SleepRSER':
        output = pn.pane.Markdown("""
                                  ### Sleep RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to fall 
                                  asleep with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones. 
                                  """, width=width)
    if RUN == 'WakeAscending':
        output = pn.pane.Markdown("""
                                  ### Wake Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """, width=width)
    if RUN == 'WakeDescending':
        output = pn.pane.Markdown("""
                                  ### Wake Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """, width=width)
    if RUN == 'WakeRSER':
        output = pn.pane.Markdown("""
                                  ### Wake RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to stay 
                                  awake with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones.
                                  """, width=width)
    return output


# Make function dependint on subject, run, and window length values
@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def dist_mot_trace(SBJ,RUN,WL_sec):
    """
    Plot the distance matrix, motion trace, and sleep stage trace in line with each other
    """
    output = (distance_matrix(SBJ,RUN,WL_sec)+motion_trace(SBJ,RUN,WL_sec)+sleep_stage_trace(SBJ,RUN,WL_sec)).cols(1)
    return output


# Display widgets player and plots
dash = pn.Column(pn.Row(pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),player),
                        pn.Column(run_order,run_description)),
          pn.Row(plot_embed3d,dist_mot_trace)).servable()

# Start gui
dash_server = dash.show(port=port_tunnel, open=False)

# +
# Stop gui
#dash_server.stop()

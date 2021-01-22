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
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory

# Load data frame of valid subjects info
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl')

# Dictionary of subject with valid runs
SubDict = {} # Empty dictionary
for i,idx in enumerate(sub_DF.index): # Iterate through each row of data frame
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    if sbj in SubDict.keys():
        SubDict[sbj].append(run) # Add run to subject list
    else: # If subject is not in dictionary yet
        SubDict[sbj] = ['All'] # Create subject element in dictionary
        SubDict[sbj].append(run) # Append run to newly created subject list

# List of all subjects
SubjectList = list(SubDict.keys())

# +
# Widgets for selecting subject run and window legth
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

@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value) # Make function dependint on subject, run, and window length widget values
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


@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value) # Make function dependint on subject, run, and window length widget values
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
    The subject, run, window lenght, and color scheme of the plot is dermined by widgets selected.
    The output of the function is the plot generated by holoviews with extension plotly.
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load data to be plotted
    # Plots with no color
    # -------------------
    if COLOR == 'No Color':
        # Select columns to be plotted, from the original data frame, and asighn to new data frame ("df")
        df = LE3D_df[['x_norm','y_norm','z_norm']].copy()
        output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm']) # Create 3D plot from 0 to "max_win" (dependent on player value)
        # Plotting options such as dimensions and size
        output = output.opts(size=5,
                             xlim=(-1,1), 
                             ylim=(-1,1), 
                             zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700)
    
    # Plots with color based on time (only for single run) or based on run (only for plotting 'All' data)
    # ---------------------------------------------------------------------------------------------------
    if COLOR == 'Time/Run':
        if RUN != 'All': # Only for single run
            # Select columns to be plotted, from the original data frame, and asighn to new data frame ("df")
            df = LE3D_df[['x_norm','y_norm','z_norm']].copy()
            df['Time [sec]'] = 2*LE3D_df.index # New column in df for time in seconds since TR=2 sec and index is TR number
            df = df.infer_objects() # Changes object type (i.e. from 'object' to 'int') if needed
            output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm'],vdims='Time [sec]') # Create 3D plot from 0 to "max_win" (dependent on player value)
            # Plotting options such as dimensions, color, and size
            output = output.opts(color='Time [sec]', # Color based on time
                                 cmap='plasma',      # Color bar spectrum
                                 colorbar=True,      # Show color bar
                                 clim=(0,get_num_tp(SBJ,RUN,WL_sec)), # Color bar lim based on number of time points in data
                                 size=5,
                                 xlim=(-1,1), 
                                 ylim=(-1,1), 
                                 zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700)
        else:
            # Select columns to be plotted, from the original data frame, and asighn to new data frame ("df")
            df = LE3D_df[['x_norm','y_norm','z_norm','Run']].copy()
            df = df.infer_objects() # Changes object type (i.e. from 'object' to 'string') if needed
            color_key  = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple','WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
            # Creates a new column in df of colors that acompany the run types, defined in the dictionary above "color_key"
            for i,idx in enumerate(df.index):
                df.loc[idx,'color'] = color_key[df.loc[idx,'Run']]
            # Creates a dictionary of the 3D plot for each run type (this is so we can add legends that tell us what run is what color)
            dict_plot={t:hv.Scatter3D(df[0:max_win].query(f'Run=="{t}"'),kdims=['x_norm','y_norm','z_norm'],vdims=['Run','color']).opts(show_legend=True,color='color',size=5,fontsize={'legend':8}) for t in df[0:max_win].Run.unique()}
            # We overlay all the plots in "dict_plot" and asighn the plotting options such as dimensions and size
            output = hv.NdOverlay(dict_plot).opts(
                              xlim=(-1,1), 
                              ylim=(-1,1), 
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=700, width=700)
            output.get_dimension('Element').label='Run '
    # Plots with color based on sleep staging from EEG
    # ------------------------------------------------
    if COLOR == 'Sleep':
        # Select columns to be plotted, from the original data frame, and asighn to new data frame ("df")
        df = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy()
        df = df.infer_objects() # Changes object type (i.e. from 'object' to 'string') if needed
        df = df.rename(columns={"Sleep Stage": "Sleep_Stage"}) # Change name becasue column name with a space is not readable for plotting
        color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        # Creates a new column in df of colors that acompany the sleep stage, defined in the dictionary above "color_key"
        for i,idx in enumerate(df.index):
                df.loc[idx,'color'] = color_key[df.loc[idx,'Sleep_Stage']]
        # Creates a dictionary of the 3D plot for each run type (this is so we can add legends that tell us what sleep stage is what color)
        dict_plot={t:hv.Scatter3D(df[0:max_win].query(f'Sleep_Stage=="{t}"'),kdims=['x_norm','y_norm','z_norm'],vdims=['Sleep_Stage','color']).opts(show_legend=True,color='color',size=5,fontsize={'legend':8}) for t in df[0:max_win].Sleep_Stage.unique()}
         # We overlay all the plots in "dict_plot" and asighn the plotting options such as dimensions and size
        output = hv.NdOverlay(dict_plot).opts(
                              xlim=(-1,1), 
                              ylim=(-1,1), 
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=700, width=700)
        output.get_dimension('Element').label='Sleep_Stage '
    # Plots with color based on motion (framewise displacement)
    # ---------------------------------------------------------
    if COLOR == 'Motion':
        # Select columns to be plotted, from the original data frame, and asighn to new data frame ("df")
        df = LE3D_df[['x_norm','y_norm','z_norm','Motion']].copy()
        max_FD = df['Motion'].max() # Maximum framewise displacement for data
        df = df.infer_objects() # Changes object type (i.e. from 'object' to 'float') if needed
        output = hv.Scatter3D(df[0:max_win], kdims=['x_norm','y_norm','z_norm'],vdims='Motion') # Create 3D plot from 0 to "max_win" (dependent on player value)
        # Plotting options such as dimensions, color, and size
        output = output.opts(opts.Scatter3D(color='Motion',
                                            cmap='jet',
                                            colorbar=True,
                                            clim=(0,max_FD),
                                            size=5,
                                            xlim=(-1,1), 
                                            ylim=(-1,1), 
                                            zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=700, width=700))
    return output


# ***
# ## Euclidean Distance Matrix

def distance_3D(x1,y1,z1,x2,y2,z2):
    """
    This fuction computes the Euclidean distance between 2 3D points.
    Where point 1 is (x1,y1,z1) and point 2 is (x2,y2,z2).
    The distance equation is sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2).
    The function outputs the distance as float number.
    """
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return dist


def distance_matrix(SBJ,RUN,WL_sec):
    """
    This fuction plots a heat map of the distnaces of each window for a given run.
    The inputs for the fuction (subject, run, and window leght) allows the user to choose what run and window leghth
    they with to plot for a given subject.
    The distance between two windows (i.e. points on the 3D plot) is computed using distance_3D() created above.
    The fuction plots the heat map using holoviews hv.HeatMap().
    The x and y axes of the plot are the two windows in which you are finding the distance.
    The z value is that distance.
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load embedding data
    data_df = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy() # New data frame of only x_norm, y_norm, and z_norm values
    dist_df = pd.DataFrame(columns=['Window1','Stage1','Window2','Stage2','Distance']) # Empty data frame to append window number and distance value
    for win1 in range(0,data_df.shape[0]): # Itterate through all windows (window 1)
        x1 = data_df.loc[win1,'x_norm'] # Assighn x value for window 1 to x1
        y1 = data_df.loc[win1,'y_norm'] # Assighn y value for window 1 to y1
        z1 = data_df.loc[win1,'z_norm'] # Assighn z value for window 1 to z1
        stage1 = data_df.loc[win1,'Sleep Stage'] # Assighn sleep stage to stage1 of window 1
        for win2 in range(0,data_df.shape[0]): # Itterate through all windows to compute distance with window 1 (window 2)
            x2 = data_df.loc[win2,'x_norm'] # Assighn x value for window 2 to x1
            y2 = data_df.loc[win2,'y_norm'] # Assighn y value for window 2 to y1
            z2 = data_df.loc[win2,'z_norm'] # Assighn z value for window 2 to z1
            stage2 = data_df.loc[win2,'Sleep Stage'] # Assighn sleep stage to stage2 of window 2
            # Append window numbers and distance to distance data frame
            dist_df.loc[len(dist_df.index)] = ['W-'+str(win1).zfill(3),stage1,'W-'+str(win2).zfill(3),stage2,distance_3D(x1,y1,z1,x2,y2,z2)] 
    output = hv.HeatMap(dist_df,kdims=['Window1','Window2'],vdims=['Distance','Stage1','Stage2']).opts(cmap='jet',colorbar=True,tools=['hover']) # Plot heat map of distances
    return output


# Example distance matrix for subject 10, run SleepRSER, and window length 30 sec
distance_matrix('sub-S10','SleepRSER',30)


# ***
# ## Plot Display

# Add a text box to describe run
@pn.depends(RunSelect.param.value)
def run_description(RUN):
    if RUN == 'All':
        output = pn.pane.Markdown("""
                                  ### All:
                                  This is the concatenated  data of all the runs. The order in which they are concatenated, 
                                  if such runs exist for that subject, are SleepAscending, Sleep Descending, Sleep RSER, 
                                  Wake Ascending, Wake Descending, and Wake RSER.
                                  """)
    if RUN == 'SleepAscending':
        output = pn.pane.Markdown("""
                                  ### Sleep Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """)
    if RUN == 'SleepDescending':
        output = pn.pane.Markdown("""
                                  ### Sleep Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to fall 
                                  asleep with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """)
    if RUN == 'SleepRSER':
        output = pn.pane.Markdown("""
                                  ### Sleep RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to fall 
                                  asleep with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones. 
                                  """)
    if RUN == 'WakeAscending':
        output = pn.pane.Markdown("""
                                  ### Wake Ascending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake. While in the scanner a repeating 10 tones in ascending order were played to
                                  the subject. The subject is not required to respond to the tones.
                                  """)
    if RUN == 'WakeDescending':
        output = pn.pane.Markdown("""
                                  ### Wake Descending:
                                  In this run the subject was placed in the scanner for around 13 minuets and asked to stay 
                                  awake with eyes closed. While in the scanner a repeating 10 tones in descending order
                                  were played to the subject. The subject is not required to respond to the tones.
                                  """)
    if RUN == 'WakeRSER':
        output = pn.pane.Markdown("""
                                  ### Wake RSER:
                                  In this run the subject was placed in the scanner for around 10 minuets and asked to stay 
                                  awake with eyes closed. In the first 5 minuets the subject was asked to rest. In the next 
                                  5 minuets the subject was asked to continue to rest and were played tones periodicaly. The
                                  subject is not required to respond to the tones.
                                  """)
    return output


# Display widgets player and plot
pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),player,pn.Row(plot_embed3d,run_description))

# ***
# ## Testing

hv.extension('bokeh')
array   = np.random.rand(20,3)
data_df = pd.DataFrame(array,columns=['x','y','z'])
sleep_stage = ['Wake','Wake','Wake','Wake','Stage 1','Stage 1','Stage 1','Stage 1','Stage 1','Stage 2','Stage 2','Stage 2','Stage 2','Stage 1','Stage 1','Stage 1','Wake','Wake','Wake','Wake']
sleep_value = [0,0,0,0,1,1,1,1,1,2,2,2,2,1,1,1,0,0,0,0]
data_df['stage']  = sleep_stage
data_df['value']  = sleep_value
data_df['window'] = data_df.index

# +
dist_df = pd.DataFrame(columns=['Window1','Stage1','Window2','Stage2','Distance'])
for win1 in range(0,data_df.shape[0]):
    x1 = data_df.loc[win1,'x']
    y1 = data_df.loc[win1,'y']
    z1 = data_df.loc[win1,'z']
    stage1 = data_df.loc[win1,'stage']
    for win2 in range(0,data_df.shape[0]):
        x2 = data_df.loc[win2,'x']
        y2 = data_df.loc[win2,'y']
        z2 = data_df.loc[win2,'z']
        stage2 = data_df.loc[win2,'stage']
        dist_df.loc[len(dist_df.index)] = [win1,stage1,win2,stage2,distance_3D(x1,y1,z1,x2,y2,z2)]
        
plot = hv.HeatMap(dist_df,kdims=['Window1','Window2'],vdims=['Distance','Stage1','Stage2']).opts(
                  cmap='jet',colorbar=True,height=500,width=650,xlabel=' ',ylabel=' ',tools=['hover'],xlim=(-2,20),ylim=(-2,20))

# +
segment_df   = pd.DataFrame(columns=['stage','start','end','start_event','end_event'])

start = 0
segment = []
for i,idx in enumerate(data_df.index):
    stage = str(data_df.loc[idx]['stage'])
    if idx == (data_df.shape[0]-1):
        segment.append(stage)
        end = start + (len(segment) - 1)
        segment_df = segment_df.append({'stage':stage, 'start':start, 'end':end,'start_event':-1, 'end_event':-1}, ignore_index=True)
    elif stage == str(data_df.loc[idx+1]['stage']):
        segment.append(stage)
    elif stage != str(data_df.loc[idx+1]['stage']):
        segment.append(stage)
        end = start + (len(segment) - 1)
        segment_df = segment_df.append({'stage':stage, 'start':start, 'end':end,'start_event':-1, 'end_event':-1}, ignore_index=True)
        start = end + 1
        segment = []
segment_df = segment_df.set_index(['stage'])


# +
wake_df = pd.DataFrame(segment_df.loc['Wake']).reset_index().drop(['stage'],axis=1)
wake_df['start'] = wake_df['start'] - 0.5
wake_df['end'] = wake_df['end'] + 0.5

stage1_df = pd.DataFrame(segment_df.loc['Stage 1']).reset_index().drop(['stage'],axis=1)
stage1_df['start'] = stage1_df['start'] - 0.5
stage1_df['end'] = stage1_df['end'] + 0.5

stage2_df = pd.DataFrame(segment_df.loc['Stage 2']).T.reset_index().drop(['index'],axis=1)
stage2_df['start'] = stage2_df['start'] - 0.5
stage2_df['end'] = stage2_df['end'] + 0.5

wake_seg_x   = hv.Segments(wake_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='orange', line_width=10)
stage1_seg_x = hv.Segments(stage1_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='yellow', line_width=10)
stage2_seg_x = hv.Segments(stage2_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='green', line_width=10)

wake_seg_y   = hv.Segments(wake_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='orange', line_width=10)
stage1_seg_y = hv.Segments(stage1_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='yellow', line_width=10)
stage2_seg_y = hv.Segments(stage2_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='green', line_width=10)

segx = wake_seg_x*stage1_seg_x*stage2_seg_x
segy = wake_seg_y*stage1_seg_y*stage2_seg_y
# -

segx*segy*plot

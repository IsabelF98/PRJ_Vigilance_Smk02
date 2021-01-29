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
import plotly.express as px
import panel as pn
import scipy
from scipy.spatial.distance import pdist, squareform
from holoviews import dim, opts
hv.extension('bokeh')
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
    The output of the function is the plot generated by plotly.
    """
    LE3D_df = load_data(SBJ,RUN,WL_sec) # Load data to be plotted
    LE3D_df = LE3D_df.infer_objects()
    if COLOR == 'No Color':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',range_x=[-1,1],range_y=[-1,1],range_z=[-1,1])
    elif COLOR == 'Time/Run':
        if RUN != 'All':
            df = pd.DataFrame(LE3D_df[0:max_win])
            output = px.scatter_3d(df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color=df.index, range_color=[0,LE3D_df.shape[0]],
                                   range_x=[-1,1],range_y=[-1,1],range_z=[-1,1])
        else:
            color_map = color_key  = {'SleepAscending':'red','SleepDescending':'orange','SleepRSER':'yellow','WakeAscending':'purple','WakeDescending':'blue','WakeRSER':'green','Inbetween Runs':'gray'}
            output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Run',range_x=[-1,1],range_y=[-1,1],range_z=[-1,1])
    elif COLOR == 'Sleep':
        color_map = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Sleep Stage',color_discrete_map=color_map,
                               range_x=[-1,1],range_y=[-1,1],range_z=[-1,1])
    elif COLOR == 'Motion':
        output = px.scatter_3d(LE3D_df[0:max_win],x='x_norm',y='y_norm',z='z_norm',color='Motion',color_continuous_scale='jet',
                               range_color=[0,LE3D_df['Motion'].max()],range_x=[-1,1],range_y=[-1,1],range_z=[-1,1])
    return output


# + jupyter={"source_hidden": true}
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
                             zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=500, width=500)
    
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
                                 zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=500, width=500)
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
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=500, width=500)
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
                              zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, margins=(5,5,5,5), height=500, width=500)
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
                                            zlim=(-1,1), aspect={'x':1,'y':1,'z':1}, camera_zoom=1, margins=(5,5,5,5), height=500, width=500))
    output = hv.render(output, backend='plotly')
    return output


# -

# ***
# ## Euclidean Distance Matrix

def stage_seg_df(segment_df,stage):
    """
    This function creates a data frame with each segment duration of a chosen sleep stage for a given run.
    The function input a segment data frame, already created, that is indexed by sleep stage and states the 
    start and end point of a stages segment in TR's called 'segment_df'.
    If a run does not have a certain stage an empty data frame is returned.
    """
    try: # Try to load stage data from segment_df
        seg_df = pd.DataFrame(segment_df.loc[stage])
    except: # If stage does not exist in data load empty data frame
        seg_df = pd.DataFrame(columns=['start','end'])
    
    if seg_df.shape[0] > 0: # If not an empyt data frame, then there is data for that stage
        if seg_df.shape[1] == 2: # If there is more then one segment of the data the data is not inverted
            seg_df = seg_df.reset_index().drop(['stage'],axis=1) # Reset index and drop 'stage' column
        else: # If there is only one segment of the data the data is inverted and must be transposed
            seg_df = seg_df.T.reset_index().drop(['index'],axis=1) # Transpose data and reset index
        # Add 0.5 to each end of segment to span entire heat map
        seg_df['start'] = seg_df['start'] - 0.5 
        seg_df['end'] = seg_df['end'] + 0.5
        
    else: # If its an empty data frame (no data for that stage exists)
        seg_df = seg_df.append({'start':0, 'end':0}, ignore_index=True) # Add "empty" data (no start of end for segment)
    
    seg_df['start_event'] = -2
    seg_df['end_event']   = -2
    return seg_df


# Make function dependint on subject, run, and window length values
@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def distance_matrix(SBJ,RUN,WL_sec):
    """
    This fuction plots a heat map of the distnaces of each window for a given run.
    The inputs for the fuction (subject, run, and window leght) allows the user to choose what run and window leghth
    they with to plot for a given subject.
    The distance between two windows (i.e. points on the 3D plot) is computed using distance_3D() created above.
    The fuction plots the heat map using holoviews hv.HeatMap().
    The x and y axes of the plot are the two windows in which you are finding the distance.
    The z value is that distance.
    A plot of the sleep staging segments are ploted along the x and y axis of the heat map using hv.Segments().
    """
    LE3D_df    = load_data(SBJ,RUN,WL_sec) # Load embedding data
    data_path  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl')
    segment_df = pd.read_pickle(data_path) 
    data_df    = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy() # New data frame of only x_norm, y_norm, and z_norm values
    
    data_array = data_df[['x_norm','y_norm','z_norm']].to_numpy() # Data as a numpy array
    dist_array = squareform(pdist(data_array, 'euclidean')) # Calculate distance matrix and rehape into one vecotr
    dist_array = np.rot90(dist_array)
    
    und_df    = stage_seg_df(segment_df,'Undetermined') # Create data frame of all undetermined segments for plotting
    wake_df   = stage_seg_df(segment_df,'Wake') # Create data frame of all wake segments for plotting
    stage1_df = stage_seg_df(segment_df,'Stage 1') # Create data frame of all stage 1 segments for plotting
    stage2_df = stage_seg_df(segment_df,'Stage 2') # Create data frame of all stage 2 segments for plotting
    stage3_df = stage_seg_df(segment_df,'Stage 3') # Create data frame of all stage 3 segments for plotting
    
    # Plot segments using hv.Segments() for each stage along x-axis
    und_seg_x    = hv.Segments(und_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='gray', line_width=20)
    wake_seg_x   = hv.Segments(wake_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='orange', line_width=20)
    stage1_seg_x = hv.Segments(stage1_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='yellow', line_width=20)
    stage2_seg_x = hv.Segments(stage2_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='green', line_width=20)
    stage3_seg_x = hv.Segments(stage3_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event']).opts(color='blue', line_width=20)
    
    # Plot segments using hv.Segments() for each stage along y-axis
    und_seg_y    = hv.Segments(und_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='gray', line_width=20)
    wake_seg_y   = hv.Segments(wake_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='orange', line_width=20)
    stage1_seg_y = hv.Segments(stage1_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='yellow', line_width=20)
    stage2_seg_y = hv.Segments(stage2_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='green', line_width=20)
    stage3_seg_y = hv.Segments(stage3_df, [hv.Dimension('start_event'), hv.Dimension('start'), 'end_event', 'end']).opts(color='blue', line_width=20)
    
    # Overlay all plots for xy-axis to create one plot for each axis
    segx = (und_seg_x*wake_seg_x*stage1_seg_x*stage2_seg_x*stage3_seg_x).opts(xlabel=' ',ylabel=' ')
    segy = (und_seg_y*wake_seg_y*stage1_seg_y*stage2_seg_y*stage3_seg_y).opts(xlabel=' ',ylabel=' ')
    
    # Plot heat map using hv.HeatMap() with hover tool
    plot = hv.Image(dist_array, bounds=(-2,-2,data_df.shape[0],data_df.shape[0])).opts(cmap='jet',xlabel=' ',ylabel=' ')
    
    # Overlay segment plots and heat map
    output = (segx*segy*plot).opts(width=500,height=400,title='Distance Martix for '+SBJ+' '+RUN+' and WL '+str(WL_sec)+' [sec]')
    return output


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
pn.Column(pn.Row(SubjSelect, RunSelect, WindowSelect, ColorSelect),
          run_description,player,
          pn.Row(plot_embed3d,distance_matrix))

# ***
# ## Testing

array   = np.random.rand(20,3)
data_df = pd.DataFrame(array,columns=['x','y','z'])
sleep_stage = ['Wake','Wake','Wake','Wake','Stage 1','Stage 1','Stage 1','Stage 1','Stage 1','Stage 2','Stage 2','Stage 2','Stage 2','Stage 1','Stage 1','Stage 1','Wake','Wake','Wake','Wake']
sleep_value = [0,0,0,0,1,1,1,1,1,2,2,2,2,1,1,1,0,0,0,0]
data_df['stage']  = sleep_stage
data_df['value']  = sleep_value
data_df['window'] = data_df.index

scatter = hv.output(hv.Scatter(data_df,kdims=['stage'],vdims=['window']).opts(color='red'), backend='bokeh')
scatter

# +
dist_df = pd.DataFrame(columns=['Window1','Stage1','Window2','Stage2','Distance'])
for win1 in range(0,data_df.shape[0]):
    stage1 = data_df.loc[win1,'stage']
    for win2 in range(0,data_df.shape[0]):
        stage2 = data_df.loc[win2,'stage']
        dist_df.loc[len(dist_df.index)] = [win1,stage1,win2,stage2,dist_array[idx]]
        
plot = hv.HeatMap(dist_df,kdims=['Window1','Window2'],vdims=['Distance','Stage1','Stage2']).opts(
                  cmap='jet',colorbar=True,height=500,width=650,xlabel=' ',ylabel=' ',tools=['hover'],xlim=(-2,20),ylim=(-2,20))
# -

dist_df.head()

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

data = load_data('sub-S30','SleepAscending',30)

# %%time
data_array = data[['x_norm','y_norm','z_norm']].to_numpy() # Data as a numpy array
dist_array = squareform(pdist(data_array, 'euclidean')).reshape(data.shape[0]**2,1) # Calculate distance matrix and rehape into one vecotr

# %%time 
data_array = data[['x_norm','y_norm','z_norm']].to_numpy() # Data as a numpy array
a = squareform(pdist(data_array, 'euclidean'))
hv.Image(a, bounds=(0,0,398,398)).opts(cmap='jet')

# +
# hv.Image?

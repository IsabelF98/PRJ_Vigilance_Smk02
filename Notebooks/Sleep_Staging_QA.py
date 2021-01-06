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

# # QA of Sleep Staging Data

import pickle
import os
import os.path as osp
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')

# +
PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
sub_DF   = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl')
sub_list = []
for i,idx in enumerate(sub_DF.index): # Iterate through each row of data frame
    sbj  = sub_DF.loc[idx]['Sbj']
    if sbj in sub_list:
        sbj = sbj
    else:
        sub_list.append(sbj)
        
SubjSelect   = pn.widgets.Select(name='Select Subject', options=sub_list, value=sub_list[0])
WindowSelect = pn.widgets.Select(name='Select Window', options=[30,46,60], value=30)


# -

def winner_takes_all(my_array):
    if np.isnan(np.sum(my_array)) == True:
        my_array[np.isnan(my_array)] = 4
    my_array = my_array.astype(int) 
    counts = np.bincount(my_array) 
    winner = np.argmax(counts)
    return winner


@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def load_EEG_data(SBJ,WL,window=False):
    PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
    sleep_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
    for RUN in ['SleepAscending','SleepDescending','SleepRSER']:
        DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        if osp.exists(DATADIR) == True:
            DATA_df  = pd.read_pickle(DATADIR)
            temp_df  = pd.DataFrame(columns=['sleep_value','sleep_stage'])
            if window == False:
                temp_df['sleep_value'] = DATA_df['sleep']
            if window == True:
                TIME = DATA_df.shape[0]
                WL_trs = int(WL/2)
                for i in range(0,TIME-WL_trs+1):
                    sleep_array  = np.array([x for x in DATA_df.loc[i:i+(WL_trs-1), 'sleep']])
                    sleep_val = winner_takes_all(sleep_array)
                    temp_df.loc[i, 'sleep_value'] = int(sleep_val)
            sleep_df = sleep_df.append(temp_df).reset_index(drop = True)
    for i,idx in enumerate(sleep_df.index):
        if sleep_df.loc[idx, 'sleep_value'] == 0:
            sleep_df.loc[idx, 'sleep_stage'] = 'Wake'
        elif sleep_df.loc[idx, 'sleep_value'] == 1:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 1'
        elif sleep_df.loc[idx, 'sleep_value'] == 2:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 2'
        elif sleep_df.loc[idx, 'sleep_value'] == 3:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 3'
        else:
            sleep_df.loc[idx, 'sleep_stage'] = 'Undetermined'
        
    wake_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        if osp.exists(DATADIR) == True:
            DATA_df  = pd.read_pickle(DATADIR)
            temp_df  = pd.DataFrame(columns=['sleep_value','sleep_stage'])
            if window == False:
                temp_df['sleep_value'] = DATA_df['sleep']
            if window == True:
                TIME = DATA_df.shape[0]
                WL_trs = int(WL/2)
                for i in range(0,TIME-WL_trs+1):
                    sleep_array  = np.array([x for x in DATA_df.loc[i:i+(WL_trs-1), 'sleep']])
                    sleep_val = winner_takes_all(sleep_array)
                    temp_df.loc[i, 'sleep_value'] = int(sleep_val)
            wake_df = wake_df.append(temp_df).reset_index(drop = True)
    for i,idx in enumerate(wake_df.index):
        if wake_df.loc[idx, 'sleep_value'] == 0:
            wake_df.loc[idx, 'sleep_stage'] = 'Wake'
        elif wake_df.loc[idx, 'sleep_value'] == 1:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 1'
        elif wake_df.loc[idx, 'sleep_value'] == 2:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 2'
        elif wake_df.loc[idx, 'sleep_value'] == 3:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 3'
        else:
            wake_df.loc[idx, 'sleep_stage'] = 'Undetermined'
    return sleep_df,wake_df


# ***
# ## Percent of Sleep Stages per Subject
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def sleep_bar_graph_TR(SBJ,WL):
    sleep_df,wake_df = load_EEG_data(SBJ,WL,window=False)
    percent_df = pd.DataFrame(columns=['Run','Sleep_Stage','Percent'])
    sleep_list = list(sleep_df['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
    wake_list = list(wake_df['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep_Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for '+SBJ)
    return output


pn.Column(SubjSelect,sleep_bar_graph_TR)

# ***
# ## Percent of Sleep Stages for All Subjects
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
sleep_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
wake_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
for SBJ in sub_list:

    for RUN in ['SleepAscending','SleepDescending','SleepRSER']:
        DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        if osp.exists(DATADIR) == True:
            DATA_df  = pd.read_pickle(DATADIR)
            temp_df  = pd.DataFrame(columns=['sleep_value','sleep_stage'])
            temp_df['sleep_value'] = DATA_df['sleep']
            sleep_df = sleep_df.append(temp_df).reset_index(drop = True)
    for i,idx in enumerate(sleep_df.index):
        if sleep_df.loc[idx, 'sleep_value'] == 0:
            sleep_df.loc[idx, 'sleep_stage'] = 'Wake'
        elif sleep_df.loc[idx, 'sleep_value'] == 1:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 1'
        elif sleep_df.loc[idx, 'sleep_value'] == 2:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 2'
        elif sleep_df.loc[idx, 'sleep_value'] == 3:
            sleep_df.loc[idx, 'sleep_stage'] = 'Stage 3'
        else:
            sleep_df.loc[idx, 'sleep_stage'] = 'Undetermined'
        
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        if osp.exists(DATADIR) == True:
            DATA_df  = pd.read_pickle(DATADIR)
            temp_df  = pd.DataFrame(columns=['sleep_value','sleep_stage'])
            temp_df['sleep_value'] = DATA_df['sleep']
            wake_df = wake_df.append(temp_df).reset_index(drop = True)
    for i,idx in enumerate(wake_df.index):
        if wake_df.loc[idx, 'sleep_value'] == 0:
            wake_df.loc[idx, 'sleep_stage'] = 'Wake'
        elif wake_df.loc[idx, 'sleep_value'] == 1:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 1'
        elif wake_df.loc[idx, 'sleep_value'] == 2:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 2'
        elif wake_df.loc[idx, 'sleep_value'] == 3:
            wake_df.loc[idx, 'sleep_stage'] = 'Stage 3'
        else:
            wake_df.loc[idx, 'sleep_stage'] = 'Undetermined'

# +
percent_df = pd.DataFrame(columns=['Run','Sleep_Stage','Percent'])
sleep_list = list(sleep_df['sleep_stage'])
for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
     percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
wake_list = list(wake_df['sleep_stage'])
for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
    percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
hv.Bars(percent_df,kdims=['Run','Sleep_Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=400,title='Sleep Stage Bar Graph for All Subjects')


# -

# ***
# ## Comparing Percent of Sleep Stages by TR vs Window
# To see if the "winner takes all" method of determining sleep stage per window we plot both bar plots on top of each other.

@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def sleep_bar_graph_WIN(SBJ,WL):
    sleep_df,wake_df = load_EEG_data(SBJ,WL,window=True)
    percent_df = pd.DataFrame(columns=['Run','Sleep_Stage','Percent'])
    sleep_list = list(sleep_df['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
    wake_list = list(wake_df['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep_Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for '+SBJ)
    return output


@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def stacked_bar_plot(SBJ,WL):
    sleep_df_WIN,wake_df_WIN = load_EEG_data(SBJ,WL,window=True)
    sleep_df_TR,wake_df_TR   = load_EEG_data(SBJ,WL,window=False)

    percent_df = pd.DataFrame(columns=['Time','Run','Sleep_Stage','Percent'])

    sleep_list_TR = list(sleep_df_TR['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['TR','Sleep',stage,(sleep_list_TR.count(stage))/len(sleep_list_TR)*100]
    wake_list_TR = list(wake_df_TR['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['TR','Wake',stage,(wake_list_TR.count(stage))/len(wake_list_TR)*100]
    
    sleep_list_WIN = list(sleep_df_WIN['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Window','Sleep',stage,(sleep_list_WIN.count(stage))/len(sleep_list_WIN)*100]
    wake_list_WIN = list(wake_df_WIN['sleep_stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Window','Wake',stage,(wake_list_WIN.count(stage))/len(wake_list_WIN)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep_Stage','Time'],group='Time').opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for '+SBJ)
    return output


pn.Column(pn.Row(SubjSelect,WindowSelect),stacked_bar_plot)

pn.Column(pn.Row(SubjSelect,WindowSelect),sleep_bar_graph_TR,sleep_bar_graph_WIN)

percent_df.set_index(['Time','Run','Sleep_Stage'])
percent_df.set_index(['Time','Run','Sleep_Stage']).hvplot.bar(by='Run')

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
        
SubjSelect = pn.widgets.Select(name='Select Subject', options=sub_list, value=sub_list[0])


# -

# ***
# ## Percent of Sleep Stages per Subject
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

@pn.depends(SubjSelect.param.value)
def sleep_bar_graph(SBJ):
    PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

    sleep_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
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
        
    wake_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
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


pn.Column(SubjSelect,sleep_bar_graph)

# ***
# ## Percent of Sleep Stages for All Subjects
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
sleep_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
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
        
    wake_df = pd.DataFrame(columns=['sleep_value','sleep_stage'])
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

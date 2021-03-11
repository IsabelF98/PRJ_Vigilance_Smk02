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

SubDict = {}
for i,idx in enumerate(sub_DF.index):
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    if sbj in SubDict.keys():
        SubDict[sbj].append(run)
    else:
        SubDict[sbj] = [run]

sub_list = list(SubDict.keys())
        
SubjSelect   = pn.widgets.Select(name='Select Subject', options=sub_list, value=sub_list[0])
RunSelect    = pn.widgets.Select(name='Select Run', options=SubDict[SubjSelect.value])
WindowSelect = pn.widgets.Select(name='Select Window', options=[30,46,60], value=30)

def update_run(event):
    RunSelect.options = SubDict[event.new]
SubjSelect.param.watch(update_run,'value')


# -

def winner_takes_all(my_array):
    if np.isnan(np.sum(my_array)) == True:
        my_array[np.isnan(my_array)] = 4
    my_array = my_array.astype(int) 
    counts = np.bincount(my_array) 
    winner = np.argmax(counts)
    return winner


def sliding_window(df,WL,fill_TR=False):
    TIME = df.shape[0]
    WL_trs = int(WL/2)
    sleep_stage_df = pd.DataFrame(columns=['time [sec]','sleep value','sleep stage'])
    for i in range(0,TIME-WL_trs+1):
        sleep_array  = np.array([x for x in df.loc[i:i+(WL_trs-1), 'sleep value']])
        sleep_val    = winner_takes_all(sleep_array)
        sleep_stage_df.loc[i, 'sleep value'] = int(sleep_val)
    if fill_TR == True:
        top_temp = pd.DataFrame(columns=['time [sec]','sleep value','sleep stage'],index=range(0,int((WL_trs-1)/2)))
        bot_temp = pd.DataFrame(columns=['time [sec]','sleep value','sleep stage'],index=range(int((TIME-WL_trs+1)+((WL_trs-1)/2)),TIME))
        sleep_stage_df = pd.concat([top_temp,sleep_stage_df,bot_temp], ignore_index=True).reset_index(drop = True)
    else:
        sleep_stage_df = sleep_stage_df
    return sleep_stage_df


def assign_sleep_stage(df):
    for i,idx in enumerate(df.index):
        if df.loc[idx, 'sleep value'] == 0:
            df.loc[idx, 'sleep stage'] = 'Wake'
        elif df.loc[idx, 'sleep value'] == 1:
            df.loc[idx, 'sleep stage'] = 'Stage 1'
        elif df.loc[idx, 'sleep value'] == 2:
            df.loc[idx, 'sleep stage'] = 'Stage 2'
        elif df.loc[idx, 'sleep value'] == 3:
            df.loc[idx, 'sleep stage'] = 'Stage 3'
        else:
            df.loc[idx, 'sleep stage'] = 'Undetermined'
    return df


def load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False):
    PRJDIR   = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
    DATADIR  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
    sleep_stage_df = pd.DataFrame(columns=['time [sec]','sleep value','sleep stage'])
    if osp.exists(DATADIR) == True:
        DATA_df  = pd.read_pickle(DATADIR)
        sleep_stage_df['sleep value'] = DATA_df['sleep']
    if window == True:
        sleep_stage_df = sliding_window(sleep_stage_df,WL,fill_TR)
    sleep_stage_df = assign_sleep_stage(sleep_stage_df)
    return sleep_stage_df


# ***
# ## Percent of Sleep Stages per Subject
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def sleep_bar_graph_TR(SBJ,WL):
    sleep_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['SleepAscending','SleepDescending','SleepRSER']:
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        sleep_df = sleep_df.append(temp_df).reset_index(drop = True)
        
    wake_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        wake_df = wake_df.append(temp_df).reset_index(drop = True)
        
    percent_df = pd.DataFrame(columns=['Run','Sleep Stage','Percent'])
    
    sleep_list = list(sleep_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
        
    wake_list = list(wake_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for '+SBJ)
    return output


pn.Column(SubjSelect,sleep_bar_graph_TR)


# ***
# ## Percent of Sleep Stages for All Subjects
# The sleep stage data is the original sleep stage data and on a TR by TR basis.

def all_sleep_data_TR(sub_list):
    WL = 0
    sleep_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for SBJ in sub_list:
        for RUN in ['SleepAscending','SleepDescending','SleepRSER']:
            DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
            temp_df = DATA_df[['sleep value','sleep stage']].copy()
            sleep_df = sleep_df.append(temp_df).reset_index(drop = True)
        
    wake_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for SBJ in sub_list:
        for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
            DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
            temp_df = DATA_df[['sleep value','sleep stage']].copy()
            wake_df = wake_df.append(temp_df).reset_index(drop = True)
        
    percent_df = pd.DataFrame(columns=['Run','Sleep_Stage','Percent'])
    
    sleep_list = list(sleep_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
        
    wake_list = list(wake_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
    return percent_df


all_percent_df = all_sleep_data_TR(sub_list)
color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
hv.Bars(all_percent_df,kdims=['Run','Sleep_Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for All Subjects')


# ***
# ## Comparing Percent of Sleep Stages by TR vs Window
# To see if the "winner takes all" method of determining sleep stage per window we plot both bar plots on top of each other.

@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def sleep_bar_graph_WIN(SBJ,WL):
    sleep_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['SleepAscending','SleepDescending','SleepRSER']: 
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=True,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        sleep_df = sleep_df.append(temp_df).reset_index(drop = True)
    
    wake_df = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=True,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        wake_df = wake_df.append(temp_df).reset_index(drop = True)
    
    percent_df = pd.DataFrame(columns=['Run','Sleep Stage','Percent'])
    
    sleep_list = list(sleep_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Sleep',stage,(sleep_list.count(stage))/len(sleep_list)*100]
        
    wake_list = list(wake_df['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Wake',stage,(wake_list.count(stage))/len(wake_list)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep Stage']).opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for All Subjects')
    return output


pn.Column(pn.Row(SubjSelect,WindowSelect),sleep_bar_graph_TR,sleep_bar_graph_WIN)


@pn.depends(SubjSelect.param.value,WindowSelect.param.value)
def stacked_bar_plot(SBJ,WL):
    sleep_df_TR = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['SleepAscending','SleepDescending','SleepRSER']: 
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        sleep_df_TR = sleep_df_TR.append(temp_df).reset_index(drop = True)
    
    wake_df_TR = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        wake_df_TR = wake_df_TR.append(temp_df).reset_index(drop = True)
        
    sleep_df_WIN = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['SleepAscending','SleepDescending','SleepRSER']: 
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=True,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        sleep_df_WIN = sleep_df_WIN.append(temp_df).reset_index(drop = True)
    
    wake_df_WIN = pd.DataFrame(columns=['sleep value','sleep stage'])
    for RUN in ['WakeAscending','WakeDescending','WakeRSER']:
        DATA_df = load_sleep_stage_data(SBJ,RUN,WL,window=True,fill_TR=False)
        temp_df = DATA_df[['sleep value','sleep stage']].copy()
        wake_df_WIN = wake_df_WIN.append(temp_df).reset_index(drop = True)

    percent_df = pd.DataFrame(columns=['Time','Run','Sleep Stage','Percent'])

    sleep_list_TR = list(sleep_df_TR['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['TR','Sleep',stage,(sleep_list_TR.count(stage))/len(sleep_list_TR)*100]
    wake_list_TR = list(wake_df_TR['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['TR','Wake',stage,(wake_list_TR.count(stage))/len(wake_list_TR)*100]
    
    sleep_list_WIN = list(sleep_df_WIN['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Window','Sleep',stage,(sleep_list_WIN.count(stage))/len(sleep_list_WIN)*100]
    wake_list_WIN = list(wake_df_WIN['sleep stage'])
    for stage in ['Wake','Stage 1','Stage 2','Stage 3', 'Undetermined']:
        percent_df.loc[len(percent_df.index)] = ['Window','Wake',stage,(wake_list_WIN.count(stage))/len(wake_list_WIN)*100]
    
    color_key  = {'Wake':'orange','Stage 1':'yellow','Stage 2':'green','Stage 3':'blue','Undetermined':'gray'}
    output = hv.Bars(percent_df,kdims=['Run','Sleep Stage','Time'],group='Time').opts(cmap=color_key,xlabel=' ',ylim=(0,100),width=800,height=350,title='Sleep Stage Bar Graph for '+SBJ)
    return output


pn.Column(pn.Row(SubjSelect,WindowSelect),stacked_bar_plot)

# TO CHANGE INDEXING
percent_df.set_index(['Time','Run','Sleep_Stage'])
percent_df.set_index(['Time','Run','Sleep_Stage']).hvplot.bar(by='Run')


# ***
# ## Comparing Sleep Sages by TR vs Windows Over Time

@pn.depends(SubjSelect.param.value,RunSelect.param.value,WindowSelect.param.value)
def sleep_stage_over_time(SBJ,RUN,WL):
    sleep_df_TR  = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
    sleep_df_WIN = load_sleep_stage_data(SBJ,RUN,WL,window=True,fill_TR=True)
    
    sleep_df_TR = sleep_df_TR[['time [sec]','sleep stage']].copy()
    sleep_df_WIN = sleep_df_WIN[['time [sec]','sleep stage']].copy()
    
    sleep_df_TR['time [sec]'] = sleep_df_TR.index*2
    sleep_df_WIN['time [sec]'] = sleep_df_WIN.index*2
    
    scatter_TR  = hv.Scatter(sleep_df_TR,vdims=['time [sec]'],kdims=['sleep stage']).opts(jitter=0.2,invert_axes=True)
    scatter_WIN = hv.Scatter(sleep_df_WIN,vdims=['time [sec]'],kdims=['sleep stage']).opts(jitter=0.2,invert_axes=True)
    
    line_TR  = hv.Curve(sleep_df_TR,vdims=['time [sec]'],kdims=['sleep stage'])
    line_WIN = hv.Curve(sleep_df_WIN,vdims=['time [sec]'],kdims=['sleep stage'])
    
    plot_title = 'Sleep Stage Over Time for '+SBJ+', '+RUN+', Window Lenght '+str(WL)+' sec'
    
    output = (line_TR*scatter_TR*line_WIN*scatter_WIN).opts(width=800, height=400, title=plot_title)
    
    return output


pn.Column(pn.Row(SubjSelect,RunSelect,WindowSelect),sleep_stage_over_time)


# ***
# ## Histogram of Sleep Segments

def count_duration(sleep_df):
    sleep_hist_df = pd.DataFrame(columns=['Subject','Run','Stage','Duration [TR]'])
    stage_segment = []
    for i,idx in enumerate(sleep_df.index):
        SBJ = sleep_df.loc[idx]['subject']
        RUN = sleep_df.loc[idx]['run']
        stage = str(sleep_df.loc[idx]['sleep stage'])
        if idx == (sleep_df.shape[0]-1):
            stage_segment.append(stage)
            sleep_hist_df = sleep_hist_df.append({'Subject':SBJ,'Run':RUN,'Stage':stage_segment[0],'Duration [TR]':len(stage_segment)}, ignore_index=True)
        elif stage == str(sleep_df.loc[idx+1]['sleep stage']):
            stage_segment.append(stage)
        elif stage != str(sleep_df.loc[idx+1]['sleep stage']):
            stage_segment.append(stage)
            sleep_hist_df = sleep_hist_df.append({'Subject':SBJ,'Run':RUN,'Stage':stage_segment[0],'Duration [TR]':len(stage_segment)}, ignore_index=True)
            stage_segment = []
    return sleep_hist_df


WL = 0
sleep_hist_df = pd.DataFrame(columns=['Subject','Run','Stage','Duration [TR]'])
for SBJ in sub_list:
    for RUN in SubDict[SBJ]:
        data_df = load_sleep_stage_data(SBJ,RUN,WL,window=False,fill_TR=False)
        sleep_df = pd.DataFrame(columns=['subject','run','sleep stage'],index=range(0,data_df.shape[0]))
        sleep_df['subject'] = SBJ
        sleep_df['run'] = RUN
        sleep_df['sleep stage'] = data_df['sleep stage']
        temp_df = count_duration(sleep_df)
        sleep_hist_df = sleep_hist_df.append(temp_df,ignore_index=True)

sleep_hist_df.to_csv(PRJDIR+'PrcsData/all/sleep_stage_duration.csv',index=False)

plot_df = sleep_hist_df[['Stage','Duration [TR]']].copy()
plot_df = plot_df.set_index(['Stage'])

wake_hist    = plot_df.loc['Wake'].value_counts().sort_index().hvplot.bar(width=1000).opts(xrotation=45,title='Frequency of Wake Duration in TRs')
stage_1_hist = plot_df.loc['Stage 1'].value_counts().sort_index().hvplot.bar(width=1000).opts(xrotation=45,title='Frequency of Stage 1 Duration in TRs')
stage_2_hist = plot_df.loc['Stage 2'].value_counts().sort_index().hvplot.bar(width=1000).opts(xrotation=45,title='Frequency of Stage 2 Duration in TRs')
stage_3_hist = plot_df.loc['Stage 3'].value_counts().sort_index().hvplot.bar(width=1000).opts(xrotation=45,title='Frequency of Stage 3 Duration in TRs')

wake_hist

stage_1_hist

stage_2_hist

stage_3_hist

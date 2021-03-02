import numpy     as np
import pandas    as pd
import sys
import os
import os.path as osp

DATADIR = '/data/SFIM_Vigilance/Data/' # Path to data directory
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
EEG_csv_file = 'all_subs_all_EEG_metrics_continuous_epochs_0_4_03-02-2021.csv' # EEG file of sleep staging data

EEG_sleep_df = pd.read_csv(DATADIR+EEG_csv_file,sep=',') # Load data frame of original EEG sleep staging data

sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl') # Load subject information data frame
SubDict = {}
for i,idx in enumerate(sub_DF.index):
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    time = sub_DF.loc[idx]['Time']
    if sbj in SubDict.keys():
        SubDict[sbj].append((run,time))
    else:
        SubDict[sbj] = [(run,time)]

SBJ_1 = sys.argv[1] # Subject number in form 'sub-S??'
SBJ_2 = SBJ_1.replace("-S","") # Subject number in form 'sub??'
RUN   = sys.argv[2] # Run name

out_path = osp.join(PRJDIR,'PrcsData',SBJ_1,'D02_Preproc_fMRI',SBJ_1+'_'+RUN+'_EEG_sleep.pkl') # File name of data being saved
#out_path = osp.join(PRJDIR,'Notebooks',SBJ_1+'_'+RUN+'_EEG_sleep.pkl')

# Empty data frame of subject spacific EEG sleep staging data
EEG_sbj_sleep_df = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds']) 

# Append subject and run data from original data frame to new data frame
for i,idx in enumerate(EEG_sleep_df.index):
    if EEG_sleep_df.loc[idx]['subject'] == SBJ_2:
        if EEG_sleep_df.loc[idx]['cond'] == RUN:
            EEG_sbj_sleep_df = EEG_sbj_sleep_df.append(pd.DataFrame(EEG_sleep_df.loc[idx]).T,ignore_index=True)

num_TR,aux = EEG_sbj_sleep_df.shape # Size of subject sleep data frame
max_TR = int(EEG_sbj_sleep_df.loc[int(num_TR)-1]['TR']) # Minimum number TR in data
min_TR = int(EEG_sbj_sleep_df.loc[0]['TR']) # Maximum number TR in data
TIME = [SubDict[SBJ_1][i][1] for i in range(0,len(SubDict[SBJ_1])) if SubDict[SBJ_1][i][0] == RUN][0] # Total number of TR's for run

# Fill in missing TRs in data bellow
if max_TR != TIME+6:
    temp_bot = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds'],index=range(0,(TIME+6)-(max_TR+1)))
    temp_bot['dataset'] = EEG_sbj_sleep_df['dataset']
    temp_bot['subject'] = EEG_sbj_sleep_df['subject']
    temp_bot['cond']    = EEG_sbj_sleep_df['cond']
    temp_bot['TR']      = range(max_TR+1,TIME+6)

# Concatinate data so all TR's are acounted for in data frame
EEG_sbj_sleep_df = pd.concat([EEG_sbj_sleep_df,temp_bot]).reset_index(drop = True)

sleep_list =[] # Emply sleep staging list
# Append sleep stage into list
#    0   = wake
#    1   = stage 1
#    2   = stage 2
#    3   = stage 3
#    NaN = undetermined stage
for i,idx in enumerate(EEG_sbj_sleep_df.index):
    if EEG_sbj_sleep_df.loc[idx, 'sleep'] == 0:
        sleep_list.append('Wake')
    elif EEG_sbj_sleep_df.loc[idx, 'sleep'] == 1:
        sleep_list.append('Stage 1')
    elif EEG_sbj_sleep_df.loc[idx, 'sleep'] == 2:
        sleep_list.append('Stage 2')
    elif EEG_sbj_sleep_df.loc[idx, 'sleep'] == 3:
        sleep_list.append('Stage 3')
    else:
        sleep_list.append('Undetermined')
        
EEG_sbj_sleep_df['stage'] = sleep_list # Add column to data of sleep stages
    
EEG_sbj_sleep_df.to_pickle(out_path) # Save data frame as pickle file in subject directory
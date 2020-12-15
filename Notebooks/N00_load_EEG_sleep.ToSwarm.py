import numpy     as np
import pandas    as pd
import sys
import os
import os.path as osp

DATADIR = '/data/SFIM_Vigilance/Data/' # Path to data directory
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
EEG_csv_file = 'all_subs_all_EEG_metrics_continuous_epochs_0_4_12-13-2020.csv' # EEG file of sleep staging data

EEG_sleep_df = pd.read_csv(DATADIR+EEG_csv_file,sep=',') # Load data frame of original EEG sleep staging data

SBJ_1 = sys.argv[1] # Subject number in form 'sub-S??'
SBJ_2 = SBJ_1.replace("-S","") # Subject number in form 'sub??'
RUN   = sys.argv[2] # Run name
out_path = osp.join(PRJDIR,'PrcsData',SBJ_1,'D02_Preproc_fMRI',SBJ_1+'_'+RUN+'_EEG_sleep.pkl') # File name of data being saved

# Empty data frame of subject spacific EEG sleep staging data
EEG_sbj_sleep_df = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds']) 

# Append subject and run data from original data frame to new data frame
for i,idx in enumerate(EEG_sleep_df.index):
    if EEG_sleep_df.loc[idx]['subject'] == SBJ_2:
        if EEG_sleep_df.loc[idx]['cond'] == RUN:
            EEG_sbj_sleep_df = EEG_sbj_sleep_df.append(pd.DataFrame(EEG_sleep_df.loc[idx]).T,ignore_index=True)
            
EEG_sbj_sleep_df.to_pickle(out_path) # Save data frame as pickle fine in subject directory
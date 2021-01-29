# The notebook will first create a date frame with the all the subject information: subject number, run name, time points in run, and region (min and max) of run in the concatinated data.
# The data frame is then saved as a .pkl file to be used in other notebooks and python files.

import os
import os.path as osp
import pandas as pd
import numpy as np

# Create and save data frame (as .pkl) for valid subjects and runs
# ----------------------------------------------------------------
# NOTE: Must run ./subject_info.sh first

# Project Directory
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# Subject and run data frame
sub_DF = pd.read_csv('./utils/subject_run.txt', delimiter=' ', header=None)
sub_DF.columns = ['Sbj','Run','Time']
sub_DF['Time Point Min'] = np.nan
sub_DF['Time Point Max'] = np.nan

# Add columns for time point regions for each run
for i,idx in enumerate(sub_DF.index):
    if idx == 0:
        sub_DF.loc[idx,'Time Point Min'] = 0
        sub_DF.loc[idx,'Time Point Max'] = sub_DF.loc[idx, 'Time'] - 1
    else:
        if sub_DF.loc[idx-1,'Sbj'] == sub_DF.loc[idx,'Sbj']:
            sub_DF.loc[idx,'Time Point Min'] = sub_DF.loc[idx-1,'Time Point Max'] +1
            sub_DF.loc[idx,'Time Point Max'] = sub_DF.loc[idx, 'Time'] -1 + sub_DF.loc[idx,'Time Point Min']
        else:
            sub_DF.loc[idx,'Time Point Min'] = 0
            sub_DF.loc[idx,'Time Point Max'] = sub_DF.loc[idx, 'Time'] - 1
            
# Save data frame as valid_run_df.pkl in Notebooks directory
sub_DF.to_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl')
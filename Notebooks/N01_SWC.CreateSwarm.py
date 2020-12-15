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

# # Create SWARM File for running N01
#
# This notebook creates the swarm file N01_SWC.SWARM.sh to run N01_SWC.ToSwarm.py (the file that creates the embeddings) for each run and window length for each subject.
# * The notebook will first create a date frame with the all the subject information: subject number, run name, time points in run, and region (min and max) of run in the concatinated data.
# * The data frame is then saved as a .pkl file to be used in other notebooks and python files.
# * A dictionary is then created of the data. Organized by subjects and its proceeding information as a tupole.
# * Finally the swarm file is created that calls the N01_SWC.ToSwarm.py for each subjects run and window length.

import os
import os.path as osp
import pandas as pd
import numpy as np

# +
# Create and save data frame (as .pkl) for valid subjects and runs
# ----------------------------------------------------------------
# NOTE: Must run ./subject_info.sh first

# Project Directory
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# Subject and run data frame
sub_DF = pd.read_csv('.utils/subject_run.txt', delimiter=' ', header=None)
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

# +
# Create dictionary of subject with valid runs
# --------------------------------------------

SubDict = {} # Empty dictionary

# Appends subject and run (including all runs) to dictionary
for i,idx in enumerate(sub_DF.index):
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    if sbj in SubDict.keys():
        SubDict[sbj].append(run)
    else:
        SubDict[sbj] = ['All']
        SubDict[sbj].append(run)

# List of subjects
SubjectList = list(SubDict.keys())

# List of window lengths in seconds
WinList = [30,46,60]

# +
# Creates N01_SWC.SWARM.sh file
# ----------------------

os.system('if [ ! -d N01_SWC.logs ]; then mkdir N01_SWC.logs; fi') # Create logs directory if doesnt already exist
os.system('echo "#swarm -f ./N01_SWC.SWARM.sh -g 32 -t 32 --time 48:00:00 --logdir ./N01_SWC.logs; watch -n 30 squeue -u fernandezis" > ./N01_SWC.SWARM.sh')
for sbj in SubjectList:
    for run in SubDict[sbj]:
        for win in WinList:
            os.system('echo "python N01_SWC.ToSwarm.py {sbj} {run} {win}" >> ./N01_SWC.SWARM.sh'.format(sbj=sbj,run=run,win=win))

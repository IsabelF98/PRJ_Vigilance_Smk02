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

# # Create SWARM File for running N00
#
# This notebook creates the swarm file N00_load_EEG_sleep.SWARM.sh to run N00_load_EEG_sleep.ToSwarm.py, the file that loads the sleep staging data for each subject and run.

import os
import os.path as osp
import pandas as pd
import numpy as np

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

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
        SubDict[sbj] = [run]

# List of subjects
SubjectList = list(SubDict.keys())

# +
# Creates N00_load_EEG_sleep.SWARM.sh file
# ----------------------------------------

os.system('if [ ! -d N00_load_EEG_sleep.logs ]; then mkdir N00_load_EEG_sleep.logs; fi') # Create logs directory if doesnt already exist
os.system('echo "#swarm -f ./N00_load_EEG_sleep.SWARM.sh -g 32 -t 32 --time 5:00:00 --logdir ./N00_load_EEG_sleep.logs; watch -n 30 squeue -u fernandezis" > ./N00_load_EEG_sleep.SWARM.sh')
for sbj in SubjectList:
    for run in SubDict[sbj]:
        os.system('echo "python N00_load_EEG_sleep.ToSwarm.py {sbj} {run}" >> ./N00_load_EEG_sleep.SWARM.sh'.format(sbj=sbj,run=run))
        

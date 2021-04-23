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

import os
import os.path as osp
import pandas as pd
import numpy as np

# +
# Create dictionary of subject with valid runs
# --------------------------------------------

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
sub_DF = pd.read_pickle(PRJDIR+'Notebooks/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs
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
# -----------------------------

os.system('if [ ! -d N01_SWC.logs ]; then mkdir N01_SWC.logs; fi') # Create logs directory if doesnt already exist
os.system('echo "#swarm -f ./N01_SWC.SWARM.sh -g 32 -t 32 --time 5:00:00 --logdir ./N01_SWC.logs; watch -n 30 squeue -u fernandezis" > ./N01_SWC.SWARM.sh')
for sbj in SubjectList:
    for run in SubDict[sbj]:
        for win in WinList:
            os.system('echo "./N01_SWC.ToSwarm.py -sbj {sbj} -run {run} -wl {win}" >> ./N01_SWC.SWARM.sh'.format(sbj=sbj,run=run,win=win))

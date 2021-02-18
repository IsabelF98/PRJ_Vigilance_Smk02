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

# # SWARM File for Extracting 4th Ventrical Signal
#
# This notebook creates a the SC02_extract_signal.SWARM.sh to execute SC02_extract_signal.sh for each subject and run.

import os
import os.path as osp
import pandas as pd

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
sub_DF = pd.read_pickle(PRJDIR+'4thVent/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

# +
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
# Create logs directory if doesnt already exist
os.system('if [ ! -d SC02_extract_signal.logs ]; then mkdir SC02_extract_signal.logs; fi')

# Call to swarm line
os.system('echo "#swarm -f ./SC02_extract_signal.SWARM.sh -g 32 -t 32 --time 5:00:00 --logdir ./SC02_extract_signal.logs; watch -n 30 squeue -u fernandezis" > ./SC02_extract_signal.SWARM.sh')

# Call to SC02_extract_signal.sh for each subject and run
for SBJ in SubjectList:
    for i,RUN_NAM in enumerate(SubDict[SBJ]):
        os.system('echo " export SBJ={SBJ} RUN_NAM={RUN_NAM} RUN_NUM={RUN_NUM}; sh ./SC02_extract_signal.sh" >> ./SC02_extract_signal.SWARM.sh'.format(SBJ=SBJ,RUN_NAM=RUN_NAM,RUN_NUM=str(i+1)))

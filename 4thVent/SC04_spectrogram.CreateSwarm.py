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

# ## SWARM File for Creating Spectrogram of 4th Ventrical Signal
#
# This notebook creates the SC04_spectrogram.SWARM.sh to execute SC04_spectrogram.py for each subject and run.

import os
import os.path as osp
import numpy as np
import pandas as pd

# +
PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
DATA_DIR = PRJDIR+'PrcsData' # Data directory

sub_DF = pd.read_pickle(PRJDIR+'4thVent/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

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
# -

# Create logs dictinary if doesn't already exist
if not osp.exists('./SC04_spectrogram.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./SC04_spectrogram.logs')

# +
# Call to swarm line
os.system('echo "#swarm -f ./SC04_spectrogram.SWARM.sh -g 16 -t 16 --partition quick,norm --time 00:20:00 --logdir ./SC04_spectrogram.logs; watch -n 30 squeue -u fernandezis" > ./SC04_spectrogram.SWARM.sh')

# Call to SC03_peridiogram.py for each subject and run
for SBJ in SubjectList:
    for RUN in SubDict[SBJ]:
        os.system('echo "export SBJ={SBJ} RUN={RUN} DATADIR={ddir}; sh ./SC04_spectrogram.sh" >> ./SC04_spectrogram.SWARM.sh'.format(SBJ=SBJ, RUN=RUN, ddir=DATA_DIR))

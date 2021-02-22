# 2/22/2021   Isabel Fernandez

# This scripts calculates the framewise displacement for each subject using the concatinated motion derivative file (motion_deriv.1D)
# FD equation FD = |dx|+|dy|+|dz|+|da|+|db|+|dc| where x, y, and z are poolar and a, b, and c are spherical

import numpy     as np
import pandas    as pd
import sys
import os
import os.path as osp

PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
SBJ = sys.argv[1] # Subject id

out_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI','framewise_displacement.1D') # Output file path and name
mot_file_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI','motion_deriv.1D') # Path to motion data for subject

mot_df   = pd.read_csv(mot_file_path,sep=' ',header=None,names=['trans_dx','trans_dy','trans_dz','rot_dx','rot_dy','rot_dz']) # Load motion data for subject

FD_df = pd.DataFrame(columns=['FD']) # Empty framewise displacement data frame
# Calculateing FD
FD_df['FD'] = abs(mot_df['trans_dx']) + abs(mot_df['trans_dy']) + abs(mot_df['trans_dz']) + abs(np.deg2rad(mot_df['rot_dx'])*50) + abs(np.deg2rad(mot_df['rot_dy'])*50) + abs(np.deg2rad(mot_df['rot_dz'])*50)

FD_df.to_csv(out_path,index=False) # Save data frame to subject directory
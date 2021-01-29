# Code to create segment data frame for distance matrix in Embedding_Plot.ipynb.
# Each row in the data frame is a different sleep stage segment in the data.
# The start window number and end window number are given as well as the sleep stage
# the segment coresponds to.

import pickle
import os
import os.path as osp
import pandas as pd
import sys
import numpy as np

# Variables for calling data
SBJ                    = sys.argv[1]
RUN                    = sys.argv[2]
WL_sec                 = int(sys.argv[3])
atlas_name             = 'Craddock_T2Level_0200'
WS_trs                 = 1
TR                     = 2.0
dim_red_method         = 'PCA'
dim_red_method_percent = 97.5
le_num_dims            = 3
le_k_NN                = 100

print('++INFO: Subject '+SBJ)
print('        Run     '+RUN)
print('        Win Len '+RUN)
print(' ')


#Path to data
PRJDIR       = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
path_datadir = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI')
data_prefix  = SBJ+'_fanaticor_'+atlas_name+'_wl'+str(WL_sec).zfill(3)+'s_ws'+str(int(WS_trs*TR)).zfill(3)+'s_'+RUN
data_path   = osp.join(path_datadir,data_prefix+'_'+dim_red_method+'_vk'+str(dim_red_method_percent)+'.le'+str(le_num_dims)+'d_knn'+str(le_k_NN).zfill(3)+'.pkl')

# Load data
LE3D_df = pd.read_pickle(data_path)
print('++INFO: Data Loaded')
print(' ') 

# Only select 'x_norm', 'y_norm', 'z_norm', and 'Sleep Stage' columns
data_df = LE3D_df[['x_norm','y_norm','z_norm','Sleep Stage']].copy()

segment_df = pd.DataFrame(columns=['stage','start','end']) 
start = 0 # Start at 0th window
segment = [] # Empty list to append all stages in a segment (should all be the same stage)
for i,idx in enumerate(data_df.index): # Iterate through data_df index
    stage = str(data_df.loc[idx]['Sleep Stage']) # Save stage at index as 'stage'
    if idx == (data_df.shape[0]-1): # If its the last index of run
        segment.append(stage) # Append stage to segment list
        end = start + (len(segment) - 1) # Last window of the segment is the lenght of the segment list plus the start number of the segment
        # Append values to segment_df
        segment_df = segment_df.append({'stage':stage, 'start':start, 'end':end}, ignore_index=True) 
    elif stage == str(data_df.loc[idx+1]['Sleep Stage']): # If the next index values stage is equal to the current stage
        segment.append(stage) # Append stage to segment list and loop again
    elif stage != str(data_df.loc[idx+1]['Sleep Stage']): # If the next index values stage is not equal to the current stage
        segment.append(stage) # Append stage to segment list
        end = start + (len(segment) - 1) # Last window of the segment is the lenght of the segment list plus the start number of the segment
        # Append values to segment_df
        segment_df = segment_df.append({'stage':stage, 'start':start, 'end':end}, ignore_index=True)
        start = end + 1 # Start of next segment is the last window of the last segment + 1
        segment = [] # New empty segment list for next segment
segment_df = segment_df.set_index(['stage']) # Set segment_df index as the sleep stage for that segment

print('++INFO: Data Frame Finished')
print(' ') 

out_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl') # File name of data being saved

segment_df.to_pickle(out_path) # Save data frame as pickle file in subject directory

print('++INFO: Data Frame saved to')
print('        '+out_path)
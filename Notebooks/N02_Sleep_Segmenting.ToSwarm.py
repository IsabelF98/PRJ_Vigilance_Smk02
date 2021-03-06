#! /usr/bin/env python

# Code to create segment data frame for distance matrix in Embedding_Plot.ipynb.
# Each row in the data frame is a different sleep stage segment in the data.
# The start window number and end window number are given as well as the sleep stage
# the segment coresponds to.

import pickle
import os
import os.path as osp
import pandas as pd
import argparse
import numpy as np

def run(args):
    # Variables for calling data
    SBJ                    = args.subject
    RUN                    = args.run
    WL_sec                 = int(args.wl)
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

    # Add 0.5 to each end of segment to span entire heat map
    segment_df['start'] = segment_df['start'] - 0.5 
    segment_df['end'] = segment_df['end'] + 0.5

    # 'start_event' and 'end_event' represent the axis along which the segments will be (-2 so it is not on top of the heat map)
    segment_df['start_event'] = -2
    segment_df['end_event']   = -2

    print('++INFO: Data Frame Finished')
    print(' ') 

    out_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl') # File name of data being saved

    segment_df.to_pickle(out_path) # Save data frame as pickle file in subject directory

    print('++INFO: Data Frame saved to')
    print('        '+out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute sleep staging segments for a given subject, run, and window length.")
    parser.add_argument("-sbj",help="subject name in sub-SXX format" ,dest="subject", type=str, required=True)
    parser.add_argument("-run",help="run name" ,dest="run", type=str, required=True)
    parser.add_argument("-wl",help="window legnth in seconds (30, 46, or 60)" ,dest="wl", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
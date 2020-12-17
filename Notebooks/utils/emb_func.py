import numpy     as np
import pandas    as pd
import xarray    as xr
import os.path as osp

def load_data(path_ts,RUN,tp_min,tp_max):
    temp_ts_df = pd.read_csv(path_ts, sep='\t', header=None)
    if RUN != 'All':
        ts_df = pd.DataFrame(temp_ts_df.loc[tp_min:tp_max])
        ts_df = ts_df.reset_index()
        ts_df = ts_df.drop('index',axis=1)
    else:
        ts_df = pd.DataFrame(temp_ts_df)
    Nacq,Nrois = ts_df.shape # Save number of time points and number of ROI's
    roi_names  = ['ROI'+str(r+1).zfill(3) for r in range(Nrois)] # ROI names (should eventually be actual names)
    ts_xr      = xr.DataArray(ts_df.values,dims=['Time [TRs]','ROIs']) # Xarray frame of data
    return ts_df,Nacq,Nrois,roi_names,ts_xr

def lapacian_dataframe(SubDict,se_X,winInfo,SBJ,RUN,TIME,WL_trs,tp_min,tp_max):
    PRJDIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
    LE3D_df      = pd.DataFrame(columns=['x','y','z','x_norm','y_norm','z_norm','Sleep Value','Sleep Stage','Motion','label'])
    LE3D_df['x'] = se_X[:,0]
    LE3D_df['y'] = se_X[:,1]
    LE3D_df['z'] = se_X[:,2]
    # Note: there is a change in scale between scikit-learn 0.19 and 0.23 when it comes to the laplacian embeddings.
    # I checked a few examples and the structure is the same, but the scale is different. To be able to represent all cases
    # on the same scale (and given that the dimensions are meaningless), I create this normalized version of the low dimensional embedding
    LE3D_df[['x_norm','y_norm','z_norm']]= LE3D_df[['x','y','z']]/LE3D_df[['x','y','z']].max()
    
    # Sleep-based data
    sleep_temp = pd.DataFrame(columns=['Sleep Value','Sleep Stage'])
    if RUN != 'All':
        sleep_file_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl')
        EEG_sleep_df    = pd.read_pickle(sleep_file_path)
    else:
        run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ]))]
        run_list.remove('All')
        EEG_sleep_df = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds'])
        for r in run_list:
            sleep_file_path    = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+r+'_EEG_sleep.pkl')
            run_sleep_df = pd.read_pickle(sleep_file_path)
            EEG_sleep_df = EEG_sleep_df.append(run_sleep_df).reset_index(drop = True)
    for i in range(0,TIME-WL_trs+1):
        sleep_list = np.array([x for x in EEG_sleep_df.loc[i:i+(WL_trs-1), 'sleep']])
        sleep_mean = np.nanmean(sleep_list)
        if np.isnan(sleep_mean) == True:
            sleep_temp.loc[i, 'Sleep Value'] = sleep_mean
        else:
            sleep_temp.loc[i, 'Sleep Value'] = int(sleep_mean)
    for i,idx in enumerate(sleep_temp.index):
        if sleep_temp.loc[idx, 'Sleep Value'] == 0:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Wake'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 1:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 1'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 2:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 2'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 3:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 3'
        else:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Undetermined'
    LE3D_df['Sleep Value'] = sleep_temp['Sleep Value']
    LE3D_df['Sleep Stage'] = sleep_temp['Sleep Stage']
    
    # Motion-based data
    mot_temp      = pd.DataFrame(columns=['Framewise Displacement'])
    mot_file_path = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Preproc_fMRI','motion_deriv.1D')
    temp_mot_df   = pd.read_csv(mot_file_path,sep=' ',header=None,names=['trans_dx','trans_dy','trans_dz','rot_dx','rot_dy','rot_dz'])
    if RUN != 'All':
        mot_df = pd.DataFrame(temp_mot_df.loc[tp_min:tp_max])
    else:
        mot_df = pd.DataFrame(temp_mot_df)
    mot_df['FD'] = abs(mot_df['trans_dx']) + abs(mot_df['trans_dy']) + abs(mot_df['trans_dz']) + abs(np.deg2rad(mot_df['rot_dx'])*50) + abs(np.deg2rad(mot_df['rot_dy'])*50) + abs(np.deg2rad(mot_df['rot_dz'])*50)
    for i in range(0,TIME-WL_trs+1):
        mot_list = np.array([x for x in mot_df.loc[i:i+(WL_trs-1), 'FD']])
        mot_mean = np.nanmean(mot_list)
        mot_temp.loc[i, 'Framewise Displacement'] = mot_mean
    LE3D_df['Motion'] = mot_temp['Framewise Displacement']
    
    # Window Names
    LE3D_df['label'] = winInfo['winNames']
    return LE3D_df
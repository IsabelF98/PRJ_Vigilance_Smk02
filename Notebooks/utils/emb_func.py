import numpy     as np
import pandas    as pd
import xarray    as xr

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

def lapacian_dataframe(se_X,winInfo,RUN):
    LE3D_df      = pd.DataFrame(columns=['x','y','z','x_norm','y_norm','z_norm','no_color_rgb','no_color_hex','time_color_rgb','label'])
    LE3D_df['x'] = se_X[:,0]
    LE3D_df['y'] = se_X[:,1]
    LE3D_df['z'] = se_X[:,2]
    # Note: there is a change in scale between scikit-learn 0.19 and 0.23 when it comes to the laplacian embeddings.
    # I checked a few examples and the structure is the same, but the scale is different. To be able to represent all cases
    # on the same scale (and given that the dimensions are meaningless), I create this normalized version of the low dimensional embedding
    LE3D_df[['x_norm','y_norm','z_norm']]= LE3D_df[['x','y','z']]/LE3D_df[['x','y','z']].max()

    # External-data based color
    LE3D_df['no_color_rgb'] = [(204,209,209) for i in range(winInfo['numWins'])]
    LE3D_df['no_color_hex'] = ['#CCD1D1' for i in range(winInfo['numWins'])]

    # Time-based color
    time_color_rbg_temp = pd.DataFrame(LE3D_df['time_color_rgb'])
    if RUN == 'All': # Color based on run
        time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ])-1)]
        color_list = [(255,87,34),(255,167,38),(255,235,59),(139,195,74),(0,188,212),(126,87,194)]
        x=0
        for i in range(len(time_list)):
            time_color_rbg_temp.loc[x:(x-1)+time_list[i]-(WL_trs-1), 'time_color_rgb'] = [color_list[i]] # color for run windows
            x=x+time_list[i]-(WL_trs-1)
            if i != len(time_list)-1:
                time_color_rbg_temp.loc[x:(x-1)+(WL_trs-1), 'time_color_rgb'] = [(204,209,209)] # color for between run windows
                x=x+(WL_trs-1)
        LE3D_df['time_color_rgb'] = time_color_rbg_temp['time_color_rgb']
    else: # Color changes over time
        time_color_rbg_temp.loc[0:244, 'time_color_rgb'] = [(n,0,0) for n in range(10,255)]
        time_color_rbg_temp.loc[245:winInfo['numWins']-1, 'time_color_rgb'] = [(255,n,n) for n in range(winInfo['numWins']-245)]
        LE3D_df['time_color_rgb'] = time_color_rbg_temp['time_color_rgb']

    # Window Names
    LE3D_df['label'] = winInfo['winNames']
    return LE3D_df

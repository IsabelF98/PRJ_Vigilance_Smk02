import numpy     as np
import pandas    as pd
import xarray    as xr
import holoviews as hv
import panel     as pn

from scipy.cluster.hierarchy import (linkage, optimal_leaf_ordering,leaves_list)
from sklearn.decomposition import PCA

def plot_fc_matrix(ts,labels, reorder='single',width=400,cmap='RdBu_r'):
    '''This function sorts FC matrix using the same algorithm as nilearn.plotting.plot_matrix
    INPUTS
    ------
    ts: ROI timeseries as pd.DataFrame
    labels: ROI names as string array
    reorder: method for re-ordering the matrix. Possible values: single, complete, average (see nilearn help for more information)
    width: size of output figure
    cmap: colormap for the connectivity matrix
    
    OUTPUTS
    -------
    hv.heatmap with the connectivity matrix
    '''
    mat     = ts.corr().values
    if reorder != False:
        linkage_matrix  = linkage(mat, method=reorder)
        ordered_linkage = optimal_leaf_ordering(linkage_matrix, mat)
        index           = leaves_list(ordered_linkage)
        # make sure labels is an ndarray and copy it
        labels = np.array(labels).copy()
        mat    = mat.copy()
        # and reorder labels and matrix
        labels = labels[index].tolist()
        mat    = mat[index, :][:, index]
    fc_matrix = pd.DataFrame(mat, index=labels,columns=labels)
    return fc_matrix.hvplot.heatmap(cmap=cmap,aspect='square').redim.range(value=(-1,1)).opts(xrotation=45, frame_width=width)

def compute_swc(ts,wl_trs,ws_trs,win_names=None,window=None):
    '''
    This function will perform the following actions:
    1) Generate windows based on length, step and TR. This means computing window onsets and offsets
    2) Generate window names if those are not provided
    3) For each sliding window:
       * extract time series for all ROIs
       * multiply by the provided window shape
       * compute connectivity matrix
       * extract top triangle
       * apply fisher-transform
       
    INPUTS
    ------
    ts: ROI timeseries in the form of a pd.DataFrame
    wl_trs: window length in number of TRs (int)
    ws_trs: window step in number of TRs (int)
    win_names: window labels as string array. If empty, labels will be generated automatically
    window: window shape to apply (np.array of length equal to wl_trs)
    
    OUTPUTS
    -------
    swc_r: sliding window connectivity matrix as Pearson's correlation (pd.Dataframe)
    swc_Z: sliding window connectivity matrix as Fisher's transform (pd.Dataframe)
    winInfo: dictionary containing window onsets, offsets, and labels.
    '''
    [Nacq,Nrois] = ts.shape
    # Create Window Information
    winInfo             = {'durInTR':int(wl_trs),'stepInTR':int(ws_trs)}
    # Computer Number of Windows
    winInfo['numWins']  = int(np.ceil((Nacq-(winInfo['durInTR']-1))/winInfo['stepInTR']))
    # Compute Window Onsets
    winInfo['onsetTRs'] = np.linspace(0,winInfo['numWins'],winInfo['numWins']+1, dtype='int')[0:winInfo['numWins']]
    winInfo['offsetTRs']= winInfo['onsetTRs'] + winInfo['durInTR']
    # Create Window Names
    if win_names is None:
        winInfo['winNames'] = ['W'+str(i).zfill(4) for i in range(winInfo['numWins'])]
    else:
        winInfo['winNames'] = win_names
    # Create boxcar window (if none provided)
    if window is None:
        window=np.ones((wl_trs,))
    # Compute SWC Matrix
    for w in range(winInfo['numWins']):
        aux_ts          = ts[winInfo['onsetTRs'][w]:winInfo['offsetTRs'][w]]
        aux_ts_windowed = aux_ts.mul(window,axis=0)
        aux_fc          = aux_ts_windowed.corr()
        sel             = np.triu(np.ones(aux_fc.shape),1).astype(np.bool)
        aux_fc_v        = aux_fc.where(sel)

        if w == 0:
            swc_r  = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
        else:
            new_df = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
            swc_r  = pd.concat([swc_r,new_df],axis=1)
    swc_Z = swc_r.apply(np.arctanh)
    return swc_r, swc_Z, winInfo

def reduce_dimensionality_pca(data,var_to_keep,sbj_id='Jonh Doe',return_graphs=True,scree_plot_width=700, scree_plot_height=500, time_series_plot_width=1000,n_comp=None, correct_sign=True):
    '''
    This function applies PCA to a matrix of representative time series per ROI, and returns a 
    new matrix of representative time series for kept PCA components.
    
    INPUTS
    ------
    data: matrix of timeseries per ROI (pd.Dataframe)
    var_to_keep: percent of variance to keep in the PCA step. This will determine how many PCA
                 components we keep for subsequent analyses.
    sbj_id:      subject_id. This is use for the title of graphs (string)
    return_graphs: a boolean to determine whether or not the function should return a visual summary
                   of this dimensionality reduction step. If True, the function will return a second output
                   in the form of an holoview plot that will contain a scree plot for variance explained, a carpet
                   plot of all kept PCA component timeseries, and graphs of timeseries for the top 5 PCA components
    scree_plot_width: width for the scree plot
    scree_plot_height: height for the scree plot
    time_series_plot_width: width of time series plot
    
    OUTPUTS
    -------
    ts_pca_df: representative timeseries for kept PCA components (pd.Dataframe)
    pca_plot: graphical depiction of PCA analysis (hv Graph). Only returned if return_graphs is True
    '''
    pca              = PCA(n_components = n_comp)
    Nacq, Nrois      = data.shape
    aux_pca          = pca.fit_transform(data.values)
    if correct_sign:
        print('++ INFO [PCA]: Sign correction (sum of coeffs positive)')
        # https://stats.stackexchange.com/questions/259414/how-does-pca-function-in-matlab-fix-the-sign-of-principal-components
        # Matlab the loading with highest absolute value is always positive
        aux_pca_sign = [-1 if i==False else 1 for i in np.equal(np.absolute(pca.components_).max(axis=1),pca.components_.max(axis=1))]
        #aux_pca_sign = [-1 if i==False else 1 for i in (pca.components_>0).sum(axis=1)>pca.components_.shape[1]/2]
        #aux_pca_sign = np.sign(pca.components_.sum(axis=1))
        aux_pca      = aux_pca * aux_pca_sign
    ts_pca_df        = pd.DataFrame(aux_pca,columns=['PC'+str(i).zfill(3) for i in range(aux_pca.shape[1])])
    ts_pca_cum_var   = 100 * np.cumsum(pca.explained_variance_ratio_)
    if n_comp is None:
        ts_pca_comp_kept = (ts_pca_cum_var < var_to_keep).sum() + 1
    else:
        ts_pca_comp_kept = n_comp
    ts_pca_df        = ts_pca_df[['PC'+str(i).zfill(3) for i in range(ts_pca_comp_kept)]]
    print('++ INFO [PCA]: Number of PCA Components to Keep: %d' % ts_pca_comp_kept)
    if return_graphs==False:
        return ts_pca_df
    else:
        # Generate Scree Plot for variance explained of PCA components
        pca_scree_plot_curve  = hv.Curve(ts_pca_cum_var).opts(xlabel='Num Components', 
                                                        ylabel='% Variance Explained', 
                                                        title='PCA on ROI Timeseries - Subject: %s' % sbj_id, 
                                                        color='black', 
                                                        line_width=.5, 
                                                        tools=['hover']).opts(width=scree_plot_height, height=scree_plot_height)
        pca_scree_plot_points = hv.Points(ts_pca_cum_var).opts(xlabel='Num Components', 
                                                         ylabel='% Variance Explained', 
                                                         title='PCA on ROI Timeseries', color='black')
        pca_scree_plot        = pca_scree_plot_curve * pca_scree_plot_points * hv.HLine(var_to_keep)
        # Generate carpet plot of PCA timeseries
        ts_pca_xr             = xr.DataArray(ts_pca_df.values,dims=['Time [TRs]','PCAs'])
        pca_carpet_plot       = ts_pca_xr.hvplot.image(cmap='gray', 
                                                       width=scree_plot_width + time_series_plot_width, 
                                                       colorbar=False, title='PCA Timeseries (carpet plot)')
        # Put graphs together, and add timeseries for top 5 components
        pca_plot = pn.Column(pn.Row(pca_scree_plot, 
                                    pn.Column(ts_pca_df['PC000'].hvplot(color='k', line_width=0.5, height=int(scree_plot_height/5), width=time_series_plot_width),
                                              ts_pca_df['PC001'].hvplot(color='k', line_width=0.5, height=int(scree_plot_height/5), width=time_series_plot_width),
                                              ts_pca_df['PC002'].hvplot(color='k', line_width=0.5, height=int(scree_plot_height/5), width=time_series_plot_width),
                                              ts_pca_df['PC003'].hvplot(color='k', line_width=0.5, height=int(scree_plot_height/5), width=time_series_plot_width),
                                              ts_pca_df['PC004'].hvplot(color='k', line_width=0.5, height=int(scree_plot_height/5), width=time_series_plot_width))),
                            pca_carpet_plot*hv.HLine(ts_pca_comp_kept))
        return ts_pca_df, pca_plot, pca
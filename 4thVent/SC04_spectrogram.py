import sys, getopt
import matplotlib
import numpy as np
import os.path as osp
import pandas as pd
from scipy.signal import spectrogram, get_window

def generate_sleep_psd(f,t,Sxx,SleepBand_BotFreq=0.03,SleepBand_TopFreq=0.07):
    # Put Spectrogram into pandas dataframe (tidy-data form)
    df = pd.DataFrame(Sxx.T, columns=f, index=t)
    df = pd.DataFrame(df.stack()).reset_index()
    df.columns=['TR','Freq','PSD']

    # Extract average timeseries of power at different bands
    sleep_band_power     = df[(df['Freq'] < SleepBand_TopFreq) & (df['Freq'] > SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    non_sleep_band_power = df[(df['Freq'] > SleepBand_TopFreq) | (df['Freq'] < SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    control_band_power   = df[(df['Freq'] > 0.1)               & (df['Freq'] < 0.2)].groupby('TR').mean()['PSD'].values
    total_power          = df.groupby('TR').mean()['PSD'].values

    # Create output dataframe, which will contain both the average power traces computed above,
    # as well as the ratio of power in sleep band to all other "control" conditions
    df3                   = pd.DataFrame(columns=['sleep','non_sleep','total','control','ratio_w_total','ratio_w_non_sleep','ratio_w_control'])
    df3['sleep']          = sleep_band_power
    df3['non_sleep']      = non_sleep_band_power
    df3['total']          = total_power
    df3['control']        = control_band_power

    df3['ratio_w_total']     = sleep_band_power/total_power
    df3['ratio_w_non_sleep'] = sleep_band_power/non_sleep_band_power
    df3['ratio_w_control']   = sleep_band_power/control_band_power
    # Set the time index for the output dataframe
    df3.index = t
    return df3

def main(argv):
    SBJ     = ''
    RUN     = ''
    WIN_LENGTH  = 30
    WIN_OVERLAP = 29
    NFFT        = 32
    SCALING     = 'density'
    DETREND     = 'linear'
    FS          = 1/2 # Sampling Frequency 1/TR

    try:
        opts,args = getopt.getopt(argv,"hs:d:r:w:p:",["subject=","run=","wdir="])
    except getopt.GetoptError:
        print ('ExtractROIs.py -s <subject> -d <run> -w <working_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('ExtractROIs.py -s <subject> -d <run> -w <working_dir>')
            sys.exit()
        elif opt in ("-s", "--subject"):
            SBJ = arg
        elif opt in ("-d", "--run"):
            RUN = arg
        elif opt in ("-w", "--wdir"):
            PRJDIR = arg
    print('++ WARNING: We assume data has a TR of 1s. Please ensure that is correct')
    print('++ =====================================================================')
    print('++ Working on %s' % SBJ)
    print(' + Run Dir       -->  %s' % RUN)
    print(' + Data Dir      -->  %s' % PRJDIR)
    print(' + Win Length    -->  %d TRs' % WIN_LENGTH)
    print(' + Win Overlap   -->  %d TRs' % WIN_OVERLAP)
    print(' + Detrend       -->  %s'     % str(DETREND))
    print(' + Sampling Freq -->  %f Hz' % FS)

    #Inputs
    #======
    RUND_Path  = osp.join(PRJDIR,SBJ,'D03_4thVent')
    roits_path = osp.join(RUND_Path,'{SBJ}.{RUN}.volreg.Signal.V4.1D'.format(SBJ=SBJ, RUN=RUN))

    print('++ Loading ROI timeseries into memory from [%s]' % roits_path)
    roits      = np.loadtxt(roits_path)

    # Compute Spectrogram
    # ===================
    print('++ Computing Spectrogram...')
    f,t,Sxx        = spectrogram(roits,FS,window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, nfft=NFFT, detrend=DETREND, mode='psd')
    spectrogram_df = pd.DataFrame(Sxx,index=f,columns=t)

    # Compute Average Power/Time in Sleep Band
    # ========================================
    print('++ Computing Summary Statistics of Power in Sleep Band...')
    band_lim_spect_df = generate_sleep_psd(f,t,Sxx)

    # Outputs
    # =======
    SPECTROGRAM_FILE   = '{SBJ}.{RUN}.volreg.Signal.V4.Spectrogram.pkl'.format(SBJ=SBJ, RUN=RUN)
    BANDLIMITED_FILE   = '{SBJ}.{RUN}.volreg.Signal.V4.Spectrogram_BandLimited.pkl'.format(SBJ=SBJ, RUN=RUN)
    SPECTROGRAM_Path   = osp.join(RUND_Path,SPECTROGRAM_FILE)
    BANDLIMITED_Path   = osp.join(RUND_Path,BANDLIMITED_FILE)

    # Save Results
    # ============
    print('++ Saving Power Spectrum do disk: [%s]' % SPECTROGRAM_Path)
    spectrogram_df.to_pickle(SPECTROGRAM_Path)

    print('++ Saving Average Power in Sleep Band: [%s]' % BANDLIMITED_Path)
    band_lim_spect_df.to_pickle(BANDLIMITED_Path)

if __name__ == "__main__":
    main(sys.argv[1:])

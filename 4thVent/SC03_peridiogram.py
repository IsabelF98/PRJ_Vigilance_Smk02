import sys, getopt
import numpy as np
import os.path as osp
import pandas as pd
from scipy.signal import welch, get_window

def main(argv):
    SBJ     = ''
    RUN     = ''
    WIN_LENGTH  = 60 #128        #60
    WIN_OVERLAP = 30 #64         #30
    NFFT        = 64 #128        #64
    SCALING     = 'density'  #'density'
    DETREND     = 'linear'   #'constant'
    FS          = 1/2 # Sampling Frequency 1/TR
    try:
        opts,args = getopt.getopt(argv,"hs:d:r:w:p:",["subject=","rundif=","wdir="])
    except getopt.GetoptError:
        print ('ExtractROIs.py -s <subject> -d <run> -w <working_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('ExtractROIs.py -s <subject> -d <run> -w <working_dir>')
            sys.exit()
        elif opt in ("-s", "--subject"):
            SBJ = arg
        elif opt in ("-d", "--rundir"):
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

    roits_path = osp.join(PRJDIR,SBJ,'D03_4thVent','${SBJ}.${RUN}.volreg.Signal.V4.1D'.format(SBJ=SBJ, RUN=RUN))
    welch_path = osp.join(PRJDIR,SBJ,'D03_4thVent','${SBJ}.${RUN}.volreg.Signal.V4.welch.pkl'.format(SBJ=SBJ, RUN=RUN))

    # Load ROI Timeseries
    # ===================
    roits = pd.read_csv(roits_path, header=None)
    # (data is now in SPC) roits = 100 * (roits -roits.mean()) / roits.mean()
    roits.columns = ['4th vent']
    print('++ INFO: Time series loaded into memory from [%s]' % roits_path)

    # Compute Peridiogram
    # ===================
    print('++ INFO: Computing Peridiogram...')
    wf, wc = welch(roits['4th vent'], fs=FS, window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, detrend=DETREND, nfft=NFFT)
    #wc     = 10*np.log10(wc)

    # Put results into a dataframe
    # ============================
    peridiogram_df = pd.DataFrame(wc,index=wf,columns=['PSD (dB/Hz)'])
    peridiogram_df.index.rename('Frequency', inplace=True)

    # Save results to disk
    # ====================
    peridiogram_df.to_pickle(welch_path)
    print('++ INFO: Peridiogram written to disk [%s]' % welch_path)

if __name__ == "__main__":
    main(sys.argv[1:])
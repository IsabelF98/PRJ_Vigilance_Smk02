# 2/18/2021   Isabel Fernandez

# This script is created to extract the 4th ventrical signal for each subject and each run
# First create file to extract signal by calculating mean of pb03.${SBJ}.r${RUN_NUM}.volreg+tlrc and running 3dcalc
# The file that the signal is extracted from is D03_4thVent/${SBJ}.${RUN_NAM}.volreg.scale.nii.gz
# Then apply mask (all/4vent_mask.all+tlrc) to signal and get 1D file of 4th ventrical signal as D03_4thVent/${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D

set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02' # Project directory
ROI_4V_PATH='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/all/4vent_mask.all+tlrc'

cd ${PRJDIR}/PrcsData/${SBJ}

# Create Signal Percent Change version
echo "++ INFO: Create SPC version of the minimally pre-processed data..."

# Calculate mean of fMRI data (pb03.${SBJ}.r${RUN_NUM}.volreg+tlrc) and save mean file as D03_4thVent/${SBJ}.${RUN_NAM}.volreg.MEAN+tlrc
3dTstat -overwrite -mean \
    -mask D02_Preproc_fMRI/mask_epi_anat.${SBJ}+tlrc \
    -prefix D03_4thVent/${SBJ}.${RUN_NAM}.volreg.MEAN+tlrc \
    D02_Preproc_fMRI/pb03.${SBJ}.r0${RUN_NUM}.volreg+tlrc
    
3dcalc -overwrite \
    -a D02_Preproc_fMRI/pb03.${SBJ}.r0${RUN_NUM}.volreg+tlrc \
    -b D03_4thVent/${SBJ}.${RUN_NAM}.volreg.MEAN+tlrc \
    -c D02_Preproc_fMRI/mask_epi_anat.${SBJ}+tlrc \
    -expr  'c * min(200, a/b*100)*step(a)*step(b)' \
    -prefix D03_4thVent/${SBJ}.${RUN_NAM}.volreg.scale.nii.gz

# Extract a series of presentative time series
3dmaskave -quiet -mask ${ROI_4V_PATH} D03_4thVent/${SBJ}.${RUN_NAM}.volreg.scale.nii.gz      > D03_4thVent/rm.${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D
3dDetrend -prefix - -polort 3 D03_4thVent/rm.${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D\' > D03_4thVent/${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D
rm D03_4thVent/rm.${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D

echo "++ INFO: Finished script. Output ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent/${SBJ}.${RUN_NAM}.volreg.Signal.V4.1D"
# Author: Javier Gonzalez-Castillo
# Date: October 30th, 2020
#
# This script creates a FB mask common to all subjects in DSET02. This mask will used in later
# stages of the analysis to only select ROIs with valid voxels in all subjects

set -e
PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

# 1. Create output directory (if needed)
# --------------------------------------
cd ${PRJDIR}/PrcsData
if [ ! -d all ]; then
    echo "++ INFO: output directory [${PRJDIR}/PrcsData/all] created."
    mkdir all
fi

# 2. Find all available full brain masks (as created via the maximally pre-processed pipeline)
# --------------------------------------------------------------------------------------------
masks=`find ${PRJDIR}/PrcsData/sub-S??/D02_Preproc_fMRI/full_mask.lowSigma.sub-S??+tlrc.HEAD | sed 's/.HEAD//g' | tr -s '\n' ' '`

# 3. Create the group-level full brain mask
# -----------------------------------------
cd ${PRJDIR}/PrcsData/all
3dbucket -overwrite -prefix full_mask.lowSigma.all.masks ${masks}
3dTstat  -overwrite -mean -prefix rm.full_mask.lowSigma.all.masks full_mask.lowSigma.all.masks+tlrc
3dcalc -overwrite -a rm.full_mask.lowSigma.all.masks+tlrc \
       -expr 'equals(a,1)' \
       -prefix full_mask.lowSigma.all
rm rm.full_mask.lowSigma.all.masks+tlrc.*

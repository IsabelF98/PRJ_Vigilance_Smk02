# 2/17/2021   Isabel Fernandez

# This script takes ALL.CSFmask.FIXgrid.fs4vent.consensus.nii created from the HCP data and resamples it for the Samikas data set 2 by using afni's 3dresample
# Output of this script is /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/all/4vent_mask.all+tlrc

set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02' # Project directory
ORIGMASK_DIR='/data/SFIMJGC_HCP7T/HCP7T/ALL' # Original maks from HCP directory
ORIG='ALL.CSFmask.FIXgrid.fs4vent.consensus.nii.gz' # Original maks name

# If original maks is not in directory then copy it here
if [ ! -d ${ORIGMASK_NAME} ]; then 
    echo "++ INFO: Creating copy of original 4th Vent maks from HCP data to here"
    cp ${ORIGMASK_DIR}/${ORIGMASK_NAME} .
fi

MASTER='full_mask.lowSigma.all' # Name of master mask for this data set
NEW='4vent_mask.all' # Name of 4th vent mask that is resampled to this data set

# Resample 4th vent mask
3dresample -master ${PRJDIR}/PrcsData/all/${MASTER}+tlrc -rmode NN -prefix ${PRJDIR}/PrcsData/all/${NEW} -input ${ORIG}

echo "++ INFO: Resampled 4th vent mask saved as /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/all/4vent_mask.all"
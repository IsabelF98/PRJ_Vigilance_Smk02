# 11/10/2020 - Isabel Fernandez
#
# This script runs @SSwarper on each individual subjects
#
# NOTES:
#
# 1) It assumes Freesurfer was run, and uses one of its outputs as input
# 2) Input is brain.finalsurfs.nii.gz, which is already skull-stripped
#

set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'
PRCS_DATA_DIR=`echo ${PRJDIR}/PrcsData`

SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`
ANAT_DATA_DIR=`echo ${SBJ_DIR}/D01_Anatomical`

echo "++ Working on Subject ${SBJ}... pre-processing anatomical data"

# Create subject directory if needed
# ----------------------------------
if [ ! -d ${SBJ_DIR} ]; then
   echo "++ INFO: New Subject directory created [${SBJ_DIR}]"
   mkdir ${SBJ_DIR}
fi

# Create directory for outputs of anatomical pre-processing if needed
# -------------------------------------------------------------------
if [ ! -d ${ANAT_DATA_DIR} ]; then 
   echo "++ INFO: New Anatomical Preprocessing directory created [${ANAT_DATA_DIR}]"
   mkdir ${ANAT_DATA_DIR}
fi

# Enter destination folder
# ------------------------
cd ${ANAT_DATA_DIR}

# @SSwarper works best with BRIK/HEAD files... we need to transform the nii files
# -------------------------------------------------------------------------------
if [ ! -e ${ANAT_DATA_DIR}/${SBJ}_Anat+orig.HEAD ]; then
   echo "++INFO: Making a copy of original anatomical dataset in BRIK/HEAD format"
   3dcopy ${PRJDIR}/Freesurfer/${SBJ}/SUMA/brain.finalsurfs.nii.gz ${SBJ}_Anat+orig
fi

# Run @SSwarper, which will compute transformation into MNI space
# ---------------------------------------------------------------
@SSwarper                                    \
   -init_skullstr_off \
   -input  ${SBJ}_Anat+orig                  \
   -base   MNI152_2009_template_SSW.nii.gz   \
   -subid  ${SBJ}                            \
   -odir   ${ANAT_DATA_DIR}                  \
   -warpscale 0.5                            \
   -verb

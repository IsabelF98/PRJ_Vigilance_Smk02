# 2/17/2021   Isabel Fernandez

# This script takes the 4th vent mask created in script 1a and multiplies it by the full brain mask of each subject to check if the vent overlays all of the subjects masks
# The output of this script is /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/${SBJ}/4vent_mask.${SBJ}+tlrc

set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02' # Project directory
ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/' # Original data directory

subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`) # List of subject
subjects=("${subjects[@]/'README'}")   # The subject directory contains a README file (not a subject ID)
subjects=("${subjects[@]/'dataset_description.json'}")   # The subject directory contains a json file (not a subject ID)
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.
num_subjects=`echo "${#subjects[@]} -1" | bc -l` # number of subjects
echo "Subjects: ${subjects[@]}" # Print subjects being processed

for SBJ in ${subjects[@]} # For each subject
do
    # Create directory for each subject for 4th vent (D03_4thVent) info if not already exists
    if [ ! -d ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent ]; then
        echo "++ INFO: Creating D03_4thVent directory in subject directory"
        mkdir ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent
    fi
    
    # Multiply 4th vent mask for all by individual subject mask and save multiplication as 4vent_mask.${SBJ}
    if [ ! -d ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent/4vent_mask.${SBJ}* ]; then
        3dcalc -a ${PRJDIR}/PrcsData/all/4vent_mask.all+tlrc -b ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/mask_epi_anat.${SBJ}+tlrc \
                -expr '(a*b)' -prefix ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent/4vent_mask.${SBJ}
        echo "++ INFO: Created 4th vent mask for ${SBJ}"
    fi
    
    # Display number of voxels in subject vent mask
    3dROIstats -mask ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent/4vent_mask.${SBJ}+tlrc -nzvoxels ${PRJDIR}/PrcsData/${SBJ}/D03_4thVent/4vent_mask.${SBJ}+tlrc

done
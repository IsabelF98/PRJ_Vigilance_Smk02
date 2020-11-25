# 11/23/2020 - Isabel Fernandez
#
# This script does a few extra pre-processing steps necessary to conduct the sliding
# window analysis. This include:
# 1) Creating a denoised dataset with filtering between 0.017 - 0.18 (for 60s windows)
# 2) Creating a denoised dataset with filtering between 0.022 - 0.18 (for 46s windows)
# 3) Creating a denoised dataset with filtering between 0.033 - 0.18 (for 30s windows)
#
set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'
SUBJECTS_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Freesurfer/'
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}") # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}") # The subject directory contains a json file. This is not a subject ID.
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo "Subjects: ${subjects[@]}"

# Create log directory if needed
if [ ! -d ./SC03_Preproc_fMRI_Additional.logs ]; then
   mkdir ./SC03_Preproc_fMRI_Additional.logs
fi

# Write top comment in Swarm file 
echo "#Creation Date: `date`" > ./SC03_Preproc_fMRI_Additional.SWARM.sh
echo "#swarm -f ./SC03_Preproc_fMRI_Additional.SWARM.sh -g 32 -t 32 --time 24:00:00 --module afni --logdir ./SC03_Preproc_fMRI_Additional.logs --sbatch \"--export AFNI_COMPRESSOR=GZIP\"" >> ./SC03_Preproc_fMRI_Additional.SWARM.sh

# Write one entry per subject in Swarm file
for i in `seq 0 1 ${num_subjects}`
do
   if [ ! -z ${subjects[i]} ]; then 
      echo "export SBJ=${subjects[i]}; sh ./SC03_Preproc_fMRI_Additional.sh" >> ./SC03_Preproc_fMRI_Additional.SWARM.sh
   fi
done

echo "++ INFO: Script finished correctly."

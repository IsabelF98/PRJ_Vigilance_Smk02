# 11/10/2020 - Isabel Fernandez
#
# Run Freesurfer recon-all in all subjects from dataset DSET01 via swarm
#
# STEPS
# -----
# 1) Run ./SC00a_Freesurfer.CreateSwarm.sh --> To create a swarm file with one entry per subject
#
# 2) Run swarm -f ./SC00a_Freesurfer.SWARM.sh -g 24 -t 24 --time 48:00:00 --logdir ./SC00a_Freesurfer.logs --module freesurfer --sbatch \"--export SUBJECTS_DIR=/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Freesurfer/\"
#    --> To send the jobs to the cluster
#
# NOTE
# ----
# This takes approx. 14 - 18 hours per subject
# This needs to be run before @SSwarper to avoid small errors in misalignment between functional data and masks extracted from freesurfer

set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'
SUBJECTS_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Freesurfer/'
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}")   # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}")   # The subject directory contains a json file. This is not a subject ID.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo "Subjects: ${subjects[@]}"

# Create log directory if needed
# ------------------------------
if [ ! -d ./SC00a_Freesurfer.logs ]; then
   mkdir ./SC00a_Freesurfer.logs
fi

# Create Fresurfer directory if needed
# ------------------------------------
if [ ! -d ../Freesurfer ]; then
   mkdir ../Freesurfer
fi

# Write top comment in Swarm file 
# -------------------------------
echo "#Creation Date: `date`" > ./SC00a_Freesurfer.SWARM.sh
echo "#swarm -f ./SC00a_Freesurfer.SWARM.sh -g 24 -t 24 --time 48:00:00 --logdir ./SC00a_Freesurfer.logs --module freesurfer --sbatch \"--export SUBJECTS_DIR=/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Freesurfer/\"" >> ./SC00a_Freesurfer.SWARM.sh
# Write one entry per subject in Swarm file
for i in `seq 0 1 ${num_subjects}`
do
   if [ ! -z ${subjects[i]} ]; then 
      echo "recon-all -all -subject ${subjects[i]} -i ${ORIG_DATA_DIR}/${subjects[i]}/ses-1/anat/${subjects[i]}_ses-1_T1w.nii" >> ./SC00a_Freesurfer.SWARM.sh
   fi
done

echo "++ INFO: Script finished correctly."
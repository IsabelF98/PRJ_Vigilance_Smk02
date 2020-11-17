# 11/10/2020 - Isabel Fernandez
#
# Run @SSwarper on Anatomical data to compute transformantion into MNI space
# This script assumes Freesurfer was already run on the data
#
# STEPS
# -----
# 1) Run ./SC01_Preproc_Anat.CreateSwarm.sh --> To create a swarm file with one entry per subject
#
# 2) Run swarm -f ./SC01_Preproc_Anat.SWARM.sh -g 32 -t 32 --partition quick,norm --logdir ./SC01_Preproc_Anat.logs
#    --> To send the jobs to the cluster
#
# NOTE
# ----
# This takes approx. 1h 30m per subject
# This needs to be run before @SSwarper to avoid small errors in misalignment between functional data and masks extracted from freesurfer
# Also, becuase we use Freesurfer for skull stripping
set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'

subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}")   # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}")   # The subject directory contains a json file. This is not a subject ID.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`

# Create Pre directory if needed
# ------------------------------------
if [ ! -d ../PrcsData ]; then
   mkdir ../PrcsData
fi

echo "Subjects: ${subjects[@]}"
echo "#Creation Date: `date`" > ./SC01_Preproc_Anat.SWARM.sh
echo "#swarm -f ./SC01_Preproc_Anat.SWARM.sh -g 32 -t 32 --partition quick,norm --logdir ./SC01_Preproc_Anat.logs" > ./SC01_Preproc_Anat.SWARM.sh
for i in `seq 0 1 ${num_subjects}`
do
  if [ ! -z ${subjects[i]} ]; then
    echo "export SBJ=${subjects[i]}; sh ./SC01_Preproc_Anat.sh" >> ./SC01_Preproc_Anat.SWARM.sh
  fi
done

# 11/10/2020 - Isabel Fernandez
#
# Run @SUMA_Make_Spec_FS in all subjects from dataset DSET01 via swarm
# This brings the output of running Freesurfer into AFNI's world for use 
# in subsequent analysis steps.
#
# STEPS
#
# 1) Run ./SC00b_Freesurfer2AFNI.CreateSwarm.sh --> To create a swarm file with one entry per subject
#
# 2) Run swarm -f ./SC00b_Freesurfer2AFNI.SWARM.sh -g 24 -t 24 --logdir ./SC00b_Freesurfer2AFNI.logs --module afni
#    --> To send the jobs to the cluster
#
# NOTE
#
# This takes approx. 30 mins per subject
set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'
SUBJECTS_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Freesurfer/'
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
ubjects=("${subjects[@]/'README'}")   # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}")   # The subject directory contains a json file. This is not a subject ID.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo "Subjects: ${subjects[@]}"

# Create log directory if needed
if [ ! -d ./SC00b_Freesurfer2AFNI.logs ]; then
   mkdir ./SC00b_Freesurfer2AFNI.logs
fi

# Write top comment in CreateSwarm File
echo "#Creation Date: `date`" > ./SC00b_Freesurfer2AFNI.SWARM.sh
echo "#swarm -f ./SC00b_Freesurfer2AFNI.SWARM.sh -g 24 -t 24 --partition quick,norm --logdir ./SC00b_Freesurfer2AFNI.logs --module afni" >> ./SC00b_Freesurfer2AFNI.SWARM.sh

# Write one entry per subject in CreateSwarm File
for i in `seq 0 1 ${num_subjects}`
do
   if [ ! -z ${subjects[i]} ]; then
      echo "@SUMA_Make_Spec_FS -sid ${subjects[i]} -NIFTI -fspath ${SUBJECTS_DIR}/${subjects[i]}" >> ./SC00b_Freesurfer2AFNI.SWARM.sh
   fi
done
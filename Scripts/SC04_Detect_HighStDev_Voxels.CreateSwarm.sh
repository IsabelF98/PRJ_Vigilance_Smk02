# Oct/30/2020 - Javier Gonzalez-Castillo
#
set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET01/'
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}")   # The subject directory contains a README file. This is not a subject ID.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo "Subjects: ${subjects[@]}"

# Create log directory if needed
if [ ! -d ./SC04_Detect_HighStDev_Voxels.logs ]; then
   mkdir ./SC04_Detect_HighStDev_Voxels.logs
fi

# Write top comment in Swarm file 
echo "#Creation Date: `date`" > ./SC04_Detect_HighStDev_Voxels.SWARM.sh
echo "#swarm -f ./SC04_Detect_HighStDev_Voxels.SWARM.sh -g 32 -t 32 --partition quick,norm --logdir ./SC04_Detect_HighStDev_Voxels.logs" >> ./SC04_Detect_HighStDev_Voxels.SWARM.sh

# Write one entry per subject in Swarm file
for i in `seq 0 1 ${num_subjects}`
do
   if [ ! -z ${subjects[i]} ]; then 
      echo "export SBJ=${subjects[i]}; sh ./SC04_Detect_HighStDev_Voxels.sh" >> ./SC04_Detect_HighStDev_Voxels.SWARM.sh
   fi
done

echo "++ INFO: Script finished correctly."

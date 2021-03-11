# 11/30/2020 - Isabel Fernandez
#
# STEPS
#
# NOTE
#
set -e

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'
SUBJECTS_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData'
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}") # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}") # The subject directory contains a json file. This is not a subject ID.
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo "Subjects: ${subjects[@]}"

# Create log directory if needed
if [ ! -d ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.logs ]; then
   mkdir ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.logs
fi

# Write top comment in Swarm file 
echo "#Creation Date: `date`" > ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.SWARM.sh
echo "#swarm -f ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.SWARM.sh -g 32 -t 32 --partition quick,norm --module afni --logdir ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.logs --sbatch \"--export AFNI_COMPRESSOR=GZIP\"" >> ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.SWARM.sh

# Write one entry per subject in Swarm file
for SBJ in ${subjects[@]}
do
   if [ ! -d ${SUBJECTS_DIR}/${SBJ}/D02_Preproc_fMRI/DXX_Tagliazucchi_2012 ]; then 
        echo "++ INFO: Folder created [${SUBJECTS_DIR}/${SBJ}/D02_Preproc_fMRI/DXX_Tagliazucchi_2012]"
        mkdir ${SUBJECTS_DIR}/${SBJ}/D02_Preproc_fMRI/DXX_Tagliazucchi_2012
   fi
   for WL in 060 046 030
   do
     echo "export SBJ=${SBJ} WL=${WL}; sh ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.sh" >> ./SC10_Tagliazucchi_2012_Extract_ROI_timeseries.SWARM.sh
   done
done

echo "++ INFO: Script finished correctly."

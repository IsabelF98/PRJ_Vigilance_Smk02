module load afni

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/' # Folder containing the original (un-preprocessed data)
PRJ_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02'

subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}") # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}") # The subject directory contains a json file. This is not a subject ID.
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.

cd ${PRJ_DIR}/Notebooks
rm subject_run.txt
rm subject_time.txt

data_types=(SleepAscending SleepDescending SleepRSER WakeAscending WakeDescending WakeRSER)

for SBJ in ${subjects[@]}
do
    if [ ${SBJ} = 'sub-S12' ]
    then
        echo "${SBJ} SleepAscending" >> ./subject_run.txt
        echo "${SBJ} SleepDescending" >> ./subject_run.txt
        echo "${SBJ} SleepRSER" >> ./subject_run.txt
    else
        for RUN in ${data_types[@]}
        do
            if [ -f "${ORIG_DATA_DIR}/${SBJ}/ses-1/func/${SBJ}_ses-1_task-${RUN}_bold.nii" ]
            then
                echo "${SBJ} ${RUN}" >> ./subject_run.txt
            fi
        done
     fi
done

for SBJ in ${subjects[@]}
do
    run_times=(`3dinfo -nt ${PRJ_DIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/pb05.${SBJ}.r0?.scale+tlrc.HEAD | awk '{print $1}' | tr -s '\n' ' '`)
    for RT in ${run_times[@]}
    do
        echo "${SBJ} ${RT}" >> ./subject_time.txt
    done
done        
# Creates a text file called subject_run.txt in utils with all subject and run info needed
module load afni

ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/' # Folder containing the original (un-preprocessed data)
PRJ_DIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02'

subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}") # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}") # The subject directory contains a json file. This is not a subject ID.
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.
num_subjects=`echo "${#subjects[@]} -1" | bc -l`
echo ${subjects[@]}

cd ${PRJ_DIR}/Notebooks
rm subject_run.txt
#rm subject_time.txt

data_types=(SleepAscending SleepDescending SleepRSER WakeAscending WakeDescending WakeRSER)
num_datatypes=`echo "${#data_types[@]} -1" | bc -l`

for i in `seq 0 1 ${num_subjects}`
do
    runs=()
    run_times=(`3dinfo -nt ${PRJ_DIR}/PrcsData/${subjects[i]}/D02_Preproc_fMRI/pb05.${subjects[i]}.r0?.scale+tlrc.HEAD | awk '{print $1}' | tr -s '\n' ' '`)
    if [ ${subjects[i]} = 'sub-S12' ]
    then
        echo "${subjects[i]} SleepAscending 412" >> ./utils/subject_run.txt
        echo "${subjects[i]} SleepDescending 412" >> ./utils/subject_run.txt
        echo "${subjects[i]} SleepRSER 295" >> ./utils/subject_run.txt
    else
        for n in `seq 0 1 ${num_datatypes}`
        do
            if [ -f "${ORIG_DATA_DIR}/${subjects[i]}/ses-1/func/${subjects[i]}_ses-1_task-${data_types[n]}_bold.nii" ]
            then
                runs+=(${data_types[n]})
            fi
        done
        num_runs=`echo "${#runs[@]} -1" | bc -l`
        for n in `seq 0 1 ${num_runs}`
            do
                echo "${subjects[i]} ${runs[n]} ${run_times[n]}" >> ./utils/subject_run.txt
        done
     fi
done
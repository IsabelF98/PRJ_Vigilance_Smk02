# 11/10/2020 - Isabel Fernandez
#
# This script will create the pre-processing scripts for DSET01 using
# afni_proc. This is the maximally pre-processing pipeline, which includes:
# 1) Head motion regressors
# 2) Physiological noise correction via compcorr
# 3) ANATICOR correction for scanner artifacts
#
# This script requires pre-processing of anatomical data with both:
# 1) Freesurfer to obtain masks for WM and lateral ventricles
# 2) @SSwarper to obtain transformations into MNI space and skull-stripping
#
# STEPS:
# 1) Run this script to generate pre-processing scripts for all subjects
# 2) Submit to the clsuter via the following swarm command:
#    swarm -f ./SC02_Preproc_fMRI.SWARM.sh -g 32 -t 32 --time 24:00:00 --logdir ./SC02_Preproc_fMRI.logs
# ============================================================================
set -e

module load afni
module load jq

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'   # Project directory: includes Scripts, Freesurfer and PrcsData folders
ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'    # Folder containing the original (un-preprocessed data)
SUBJECTS_DIR=` echo ${PRJDIR}/Freesurfer/`           # Folder with Freesurfer results

# Initialize Swarm File
# ---------------------
echo "#Creation Time: `date`" > ./SC02_Preproc_fMRI.SWARM.sh
echo "#swarm -f ./SC02_Preproc_fMRI.SWARM.sh -g 32 -t 32 --time 24:00:00 --logdir ./SC02_Preproc_fMRI.logs" >> ./SC02_Preproc_fMRI.SWARM.sh

# Create log directory if needed (for swarm files)
# ------------------------------------------------
if [ ! -d SC02_Preproc_fMRI.logs ]; then 
   mkdir SC02_Preproc_fMRI.logs
fi

# Create directory for all fMRI data processing files per subject if needed
# -------------------------------------------------------------------------
if [ ! -d SC02_Preproc_fMRI ]; then 
   mkdir SC02_Preproc_fMRI
fi

# Get list of subjects (assumed @SSwarper scripts have been run)
# --------------------------------------------------------------
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}")   # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}")   # The subject directory contains a json file. This is not a subject ID.
#echo 'Number of subjects: '${#subjects[@]}
#echo 'Subjects: '${subjects[@]}
#echo ' '

# Get list of data names
# ----------------------
data_types=(SleepAscending SleepDescending SleepRSER WakeAscending WakeDescending WakeRSER)

# Copy and process all fMRI data
# ------------------------------
for SBJ in ${subjects[@]}
do
  ANAT_PROC_DIR=`echo ${PRJDIR}PrcsData/${SBJ}/D01_Anatomical`
  FMRI_ORIG_DIR=`echo ${PRJDIR}PrcsData/${SBJ}/D00_OriginalData`
  OUT_DIR=`echo ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI`
  
  # Create D00_OriginalData if needed
  if [ ! -d ${FMRI_ORIG_DIR} ]; then
     mkdir ${FMRI_ORIG_DIR}
  fi
  
  # Empty list of data paths
  DATA_PATHS=''
  
  for suffix in ${data_types[@]}
  do
     if [ -f "${ORIG_DATA_DIR}/${SBJ}/ses-1/func/${SBJ}_ses-1_task-${suffix}_bold.nii" ]; then
       # Copy data to D00_OriginalData in BRIK/HEAD format
       3dcopy -overwrite ${ORIG_DATA_DIR}/${SBJ}/ses-1/func/${SBJ}_ses-1_task-${suffix}_bold.nii ${FMRI_ORIG_DIR}/${SBJ}_${suffix}+orig
       # Add new path to list
       DATA_PATHS=`echo $DATA_PATHS ${FMRI_ORIG_DIR}/${SBJ}_${suffix}+orig`
       # Change slice time in .nii file header
       slice_timing=`jq '.SliceTiming' ${ORIG_DATA_DIR}/${SBJ}/ses-1/func/${SBJ}_ses-1_task-${suffix}_bold.json | awk '{print $1}' | sed '/]/d' | sed '/\[/d' | tr -s '\n' ' '`
       3drefit -Tslices $slice_timing ${FMRI_ORIG_DIR}/${SBJ}_${suffix}+orig
       3dinfo -slice_timing ${FMRI_ORIG_DIR}/${SBJ}_${suffix}+orig
    fi
  done
  
  cd ${FMRI_ORIG_DIR}
  # Run afni_proc.py to generate the pre-processing script for this particular run
  afni_proc.py                                                                                \
             -subj_id ${SBJ}                                                                  \
             -blocks despike tshift align tlrc volreg blur mask scale regress                 \
             -radial_correlate_blocks tcat volreg                                             \
             -copy_anat ${ANAT_PROC_DIR}/anatSS.${SBJ}.nii                                    \
             -anat_has_skull no                                                               \
             -anat_follower anat_w_skull anat ${ANAT_PROC_DIR}/anatUAC.${SBJ}.nii             \
             -anat_follower_ROI aaseg  anat ${SUBJECTS_DIR}/${SBJ}/SUMA/aparc.a2009s+aseg.nii \
             -anat_follower_ROI aeseg  epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/aparc.a2009s+aseg.nii \
             -anat_follower_ROI FSvent epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/fs_ap_latvent.nii.gz  \
             -anat_follower_ROI FSWe   epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/fs_ap_wm.nii.gz       \
             -anat_follower_erode FSvent FSWe                                                 \
             -tcat_remove_first_trs 5                                                         \
             -dsets $DATA_PATHS                                                               \
             -align_opts_aea -cost lpc+ZZ -giant_move -check_flip                             \
             -tlrc_base MNI152_2009_template_SSW.nii.gz                                       \
             -tlrc_NL_warp                                                                    \
             -tlrc_NL_warped_dsets   ${ANAT_PROC_DIR}/anatQQ.${SBJ}.nii                       \
                   ${ANAT_PROC_DIR}/anatQQ.${SBJ}.aff12.1D                                    \
                   ${ANAT_PROC_DIR}/anatQQ.${SBJ}_WARP.nii                                    \
             -volreg_align_to first                                                           \
             -volreg_align_e2a                                                                \
             -volreg_tlrc_warp                                                                \
             -volreg_warp_dxyz 3                                                              \
             -blur_size 6.0                                                                   \
             -mask_epi_anat yes                                                               \
             -regress_opts_3dD -jobs 32                                                       \
             -regress_motion_per_run                                                          \
             -regress_ROI_PC FSvent 3                                                         \
             -regress_ROI_PC_per_run FSvent                                                   \
             -regress_make_corr_vols aeseg FSvent                                             \
             -regress_anaticor_fast                                                           \
             -regress_anaticor_label FSWe                                                     \
             -regress_censor_motion 0.25                                                      \
             -regress_censor_outliers 0.05                                                    \
             -regress_apply_mot_types demean deriv                                            \
             -regress_est_blur_epits                                                          \
             -regress_est_blur_errts                                                          \
             -regress_bandpass 0.01 0.1                                                       \
             -regress_polort 5                                                                \
             -regress_run_clustsim no                                                         \
             -html_review_style pythonic                                                      \
             -out_dir ${OUT_DIR}                                                              \
             -script  SC02_Preproc_fMRI.${SBJ}.sh                                             \
             -volreg_compute_tsnr yes                                                         \
             -regress_compute_tsnr yes                                                        \
             -regress_make_cbucket yes                                                        \
             -scr_overwrite
  
  # Make correction to created script: use linear interpolation instead of zeros for censored datapoints
  # I need to do this post-hoc becuase afni_proc does not seem to have an option for this. 
  sed -i 's/-cenmode ZERO/-cenmode NTRP/g' SC02_Preproc_fMRI.${SBJ}.sh
   
  # Move newly created processing script to the scripts folder for this step.
  mv ${FMRI_ORIG_DIR}/SC02_Preproc_fMRI.${SBJ}.sh ${PRJDIR}/Scripts/SC02_Preproc_fMRI/

  # Add line for this subject to the Swarm file
  echo "module load afni; tcsh -xef ./SC02_Preproc_fMRI/SC02_Preproc_fMRI.${SBJ}.sh 2>&1 | tee ./SC02_Preproc_fMRI/output.SC02_Preproc_fMRI.${SBJ}.txt" >> ${PRJDIR}/Scripts/SC02_Preproc_fMRI.SWARM.sh
done

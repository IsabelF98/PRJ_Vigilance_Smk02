# Vigilance - Samika's Data Set 2

This repository containst scripts to process Samika's data set 2. 

## Anatomical Processing

This part of the analysis uses Freesurfer and AFNI. Code to run this part of the analysis is available as bash scripts.

1. Run _Freesurfer_ in anatomical data `SC00a_Freesurfer`
   * Skull-stripped version of anatomical
   * Ventricles mask (needed for CompCorr / Physiological noise removal)
   * White matter mask (needed for ANATICOR / Scanner instability confounds removal)
   * Potentially also for GM ribbon mask 

2. Run AFNI's _@SUMA_Make_Spec_FS_ to convert Freesurfer outputs into AFNI's format `SC00b_Freesurfer2AFNI`
   * SUMA folder within Freesurfer SUBJECTS_DIR is the main output

3. Run AFNI's _@SSwarper_ to compute non-linear transformation into MNI space `SC01_Preproc_Anat`

## Functional Preprocessing

This part of the analysis uses AFNI. Code to run this part of the analysis is available as bash scripts.

Functional pre-processing is accomplished with AFNI's afni_proc.py program, followed by a series of additional steps that are specific to sliding window correlation analyses. 

1. AFNI's afni_proc.py `SC02_Preproc_fMRI`
   * Spike Removal
   * Time Shift Correction (ensure TR and slice timing information is available in file headers)
   * Compute alignment between Anatomical and EPI data
   * Estimate Head Motion
   * Bring Motion corrected data into MNI Space
   * Spatial Smoothing (FWHM = 4mm)
   * Compute Full Brain Masks
   * Scale timeseries to signal percent change
   * Nuisance regression: motion, first derivative of motion, 3 PCA's from ventricles (_CompCorr_), local white matter (_ANATICOR_), bandpass filtering (0.01 - 0.1Hz)
   * Generate Quality Control HTML reports
  
2. Additional pre-processing `SC03_Preproc_fMRI_Additional`
   * Generate pre-processed timeseries with other filtering schemes
   * For 60s sliding windows [0.017 - 0.18 Hz]
   * For 46s sliding windows [0.022 - 0.18 Hz]
   * For 30s sliding windows [0.033 - 0.18 Hz]
  
3. Find voxels with excessive variance (i.e., big vasculature) `SC04_Detect_HighStDev_Voxels`
   * Find voxels with stdev > 3
   * Remove such voxels from final full brain mask
  
4. Create group-level full brain mask. `SC05_Create_Group_FBMask`

   This is a mask that contains "good" voxels for every subject. A "good" voxel is a voxel that it is within the imaged field of view of all subjects and that has no large standard deviation. This final mask will be used to decide which ROIs from a given atlas are used in subsequent analyses. This is necessary because you may have an atlas with a coverage that goes beyond that of our data, and it would be errorneous to include ROIs with no data in the computation of functional connectivity.
   * Group Level mask located in *PRJDIR/PrcsData/all/full_mask.lowSigma.all+tlrc*
   
5. Find ROIs with sufficient voxels within the group-level imaging FOV `SC06_ExtractROIs_in_GroupFOV`

   This script takes an atlas, and for each ROI it checks whether or not the ROI would still have 10 voxels after being masked with the group-level FOV mask generated in step (4). This script usually results in the drop of ventral ROIs that are not in the imaging FOV, or that are not inside the FOV mask due to excessive drop-out. 
   
   * Final atlas will be in *PRJDIR/PrcsData/all/all.$ATLAS_NAME.MNI+tlrc*
   * Information about final ROI size and which ROIs are discarded is available at: *PRJDIR/PrcsData/all/all.$ATLAS_NAME.MNI.txt*
   
6. Bring corrected atlas to final format for ROI series extraction `SC07_AdaptAtlasToSubject`
   * Mask the atlas with the subject specific FOV mask (in the future this could be changed to a GM ribbon)
   * Ensure ROI IDs are contiguous (e.g., if ROI 7 is removed, then it sould be 6 then 7, not 6 then 8)
   * Create a new version of the atlas, where the first sub-brick contains the whole atlas, and then every successive sub-brick contains each individual ROI
   * Output file: *PRJDIR/PrcsData/$SBJ/D02_Preproc_fMRI/$SBJ.$ATLASID.lowSigma+tlrc*

7. Extract representative ROI per timeseries
   * Instead of the mean, we use as representative timeseries the first PCA component.
   
## Sliding Window Correlation Analysis & Laplacian Embeddings

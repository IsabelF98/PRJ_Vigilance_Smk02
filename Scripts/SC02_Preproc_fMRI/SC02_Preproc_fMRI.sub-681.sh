#!/bin/tcsh -xef

echo "auto-generated by afni_proc.py, Tue Nov  3 14:01:43 2020"
echo "(version 7.12, April 14, 2020)"
echo "execution started: `date`"

# to execute via tcsh: 
#   tcsh -xef SC02_Preproc_fMRI.sub-681.sh |& tee output.SC02_Preproc_fMRI.sub-681.sh
# to execute via bash: 
#   tcsh -xef SC02_Preproc_fMRI.sub-681.sh 2>&1 | tee output.SC02_Preproc_fMRI.sub-681.sh

# =========================== auto block: setup ============================
# script setup

# take note of the AFNI version
afni -ver

# check that the current AFNI version is recent enough
afni_history -check_date 27 Jun 2019
if ( $status ) then
    echo "** this script requires newer AFNI binaries (than 27 Jun 2019)"
    echo "   (consider: @update.afni.binaries -defaults)"
    exit
endif

# the user may specify a single subject to run with
if ( $#argv > 0 ) then
    set subj = $argv[1]
else
    set subj = sub-681
endif

# assign output directory name
set output_dir = /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//PrcsData/sub-681/D02_Preproc_fMRI

# verify that the results directory does not yet exist
if ( -d $output_dir ) then
    echo output dir "$subj.results" already exists
    exit
endif

# set list of runs
set runs = (`count -digits 2 1 1`)

# create results and stimuli directories
mkdir $output_dir
mkdir $output_dir/stimuli

# copy anatomy to results dir
3dcopy                                                                                          \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatSS.sub-681.nii \
    $output_dir/anatSS.sub-681

# copy anatomical follower datasets into the results dir
3dcopy                                                                                           \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatUAC.sub-681.nii \
    $output_dir/copy_af_anat_w_skull
3dcopy                                                                                           \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/Freesurfer/sub-681/SUMA/aparc.a2009s+aseg.nii       \
    $output_dir/copy_af_aaseg
3dcopy                                                                                           \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/Freesurfer/sub-681/SUMA/aparc.a2009s+aseg.nii       \
    $output_dir/copy_af_aeseg
3dcopy                                                                                           \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/Freesurfer/sub-681/SUMA/fs_ap_latvent.nii.gz        \
    $output_dir/copy_af_FSvent
3dcopy                                                                                           \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/Freesurfer/sub-681/SUMA/fs_ap_wm.nii.gz             \
    $output_dir/copy_af_FSWe

# copy external -tlrc_NL_warped_dsets datasets
3dcopy                                                                                               \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681.nii      \
    $output_dir/anatQQ.sub-681
3dcopy                                                                                               \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681.aff12.1D \
    $output_dir/anatQQ.sub-681.aff12.1D
3dcopy                                                                                               \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681_WARP.nii \
    $output_dir/anatQQ.sub-681_WARP.nii

# ============================ auto block: tcat ============================
# apply 3dTcat to copy input dsets to results dir,
# while removing the first 5 TRs
3dTcat -prefix $output_dir/pb00.$subj.r01.tcat                   \
    /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D00_OriginalData/sub-681_ses-sleep-task-spatatt+orig'[5..$]'

# and make note of repetitions (TRs) per run
set tr_counts = ( 1286 )

# -------------------------------------------------------
# enter the results directory (can begin processing data)
cd $output_dir


# ---------------------------------------------------------
# data check: compute correlations with spherical ~averages
@radial_correlate -nfirst 0 -do_clean yes -rdir radcor.pb00.tcat \
                  pb00.$subj.r*.tcat+orig.HEAD

# ========================== auto block: outcount ==========================
# data check: compute outlier fraction for each volume
touch out.pre_ss_warn.txt
foreach run ( $runs )
    3dToutcount -automask -fraction -polort 18 -legendre                    \
                pb00.$subj.r$run.tcat+orig > outcount.r$run.1D

    # censor outlier TRs per run, ignoring the first 0 TRs
    # - censor when more than 0.05 of automask voxels are outliers
    # - step() defines which TRs to remove via censoring
    1deval -a outcount.r$run.1D -expr "1-step(a-0.05)" > rm.out.cen.r$run.1D

    # outliers at TR 0 might suggest pre-steady state TRs
    if ( `1deval -a outcount.r$run.1D"{0}" -expr "step(a-0.4)"` ) then
        echo "** TR #0 outliers: possible pre-steady state TRs in run $run" \
            >> out.pre_ss_warn.txt
    endif
end

# catenate outlier counts into a single time series
cat outcount.r*.1D > outcount_rall.1D

# catenate outlier censor files into a single time series
cat rm.out.cen.r*.1D > outcount_${subj}_censor.1D

# ================================ despike =================================
# apply 3dDespike to each run
foreach run ( $runs )
    3dDespike -NEW -nomask -prefix pb01.$subj.r$run.despike \
        pb00.$subj.r$run.tcat+orig
end

# ================================= tshift =================================
# time shift data so all slice timing is the same 
foreach run ( $runs )
    3dTshift -tzero 0 -quintic -prefix pb02.$subj.r$run.tshift \
             -tpattern seq-z                                   \
             pb01.$subj.r$run.despike+orig
end

# --------------------------------
# extract volreg registration base
3dbucket -prefix vr_base pb02.$subj.r01.tshift+orig"[0]"

# ================================= align ==================================
# for e2a: compute anat alignment transformation to EPI registration base
# (new anat will be current anatSS.sub-681+orig)
align_epi_anat.py -anat2epi -anat anatSS.sub-681+orig \
       -suffix _al_junk                               \
       -epi vr_base+orig -epi_base 0                  \
       -epi_strip 3dAutomask                          \
       -anat_has_skull no                             \
       -cost lpc+ZZ -giant_move -check_flip           \
       -volreg off -tshift off

# ================================== tlrc ==================================

# nothing to do: have external -tlrc_NL_warped_dsets

# warped anat     : anatQQ.sub-681+tlrc
# affine xform    : anatQQ.sub-681.aff12.1D
# non-linear warp : anatQQ.sub-681_WARP.nii

# ================================= volreg =================================
# align each dset to base volume, to anat, warp to tlrc space

# verify that we have a +tlrc warp dataset
if ( ! -f anatQQ.sub-681+tlrc.HEAD ) then
    echo "** missing +tlrc warp dataset: anatQQ.sub-681+tlrc.HEAD" 
    exit
endif

# register and warp
foreach run ( $runs )
    # register each volume to the base image
    3dvolreg -verbose -zpad 1 -base vr_base+orig                          \
             -1Dfile dfile.r$run.1D -prefix rm.epi.volreg.r$run           \
             -cubic                                                       \
             -1Dmatrix_save mat.r$run.vr.aff12.1D                         \
             pb02.$subj.r$run.tshift+orig

    # create an all-1 dataset to mask the extents of the warp
    3dcalc -overwrite -a pb02.$subj.r$run.tshift+orig -expr 1             \
           -prefix rm.epi.all1

    # catenate volreg/epi2anat/tlrc xforms
    cat_matvec -ONELINE                                                   \
               anatQQ.sub-681.aff12.1D                                    \
               anatSS.sub-681_al_junk_mat.aff12.1D -I                     \
               mat.r$run.vr.aff12.1D > mat.r$run.warp.aff12.1D

    # apply catenated xform: volreg/epi2anat/tlrc/NLtlrc
    # then apply non-linear standard-space warp
    3dNwarpApply -master anatQQ.sub-681+tlrc -dxyz 3                      \
                 -source pb02.$subj.r$run.tshift+orig                     \
                 -nwarp "anatQQ.sub-681_WARP.nii mat.r$run.warp.aff12.1D" \
                 -prefix rm.epi.nomask.r$run

    # warp the all-1 dataset for extents masking 
    3dNwarpApply -master anatQQ.sub-681+tlrc -dxyz 3                      \
                 -source rm.epi.all1+orig                                 \
                 -nwarp "anatQQ.sub-681_WARP.nii mat.r$run.warp.aff12.1D" \
                 -interp cubic                                            \
                 -ainterp NN -quiet                                       \
                 -prefix rm.epi.1.r$run

    # make an extents intersection mask of this run
    3dTstat -min -prefix rm.epi.min.r$run rm.epi.1.r$run+tlrc
end

# make a single file of registration params
cat dfile.r*.1D > dfile_rall.1D

# ----------------------------------------
# create the extents mask: mask_epi_extents+tlrc
# (this is a mask of voxels that have valid data at every TR)
# (only 1 run, so just use 3dcopy to keep naming straight)
3dcopy rm.epi.min.r01+tlrc mask_epi_extents

# and apply the extents mask to the EPI data 
# (delete any time series with missing data)
foreach run ( $runs )
    3dcalc -a rm.epi.nomask.r$run+tlrc -b mask_epi_extents+tlrc           \
           -expr 'a*b' -prefix pb03.$subj.r$run.volreg
end

# warp the volreg base EPI dataset to make a final version
cat_matvec -ONELINE                                                       \
           anatQQ.sub-681.aff12.1D                                        \
           anatSS.sub-681_al_junk_mat.aff12.1D -I  > mat.basewarp.aff12.1D

3dNwarpApply -master anatQQ.sub-681+tlrc -dxyz 3                          \
             -source vr_base+orig                                         \
             -nwarp "anatQQ.sub-681_WARP.nii mat.basewarp.aff12.1D"       \
             -prefix final_epi_vr_base

# create an anat_final dataset, aligned with stats
3dcopy anatQQ.sub-681+tlrc anat_final.$subj

# record final registration costs
3dAllineate -base final_epi_vr_base+tlrc -allcostX                        \
            -input anat_final.$subj+tlrc |& tee out.allcostX.txt

# --------------------------------------
# create a TSNR dataset, just from run 1
3dTstat -mean -prefix rm.signal.vreg.r01 pb03.$subj.r01.volreg+tlrc
3dDetrend -polort 18 -prefix rm.noise.det -overwrite                      \
    pb03.$subj.r01.volreg+tlrc
3dTstat -stdev -prefix rm.noise.vreg.r01 rm.noise.det+tlrc
3dcalc -a rm.signal.vreg.r01+tlrc                                         \
       -b rm.noise.vreg.r01+tlrc                                          \
       -c mask_epi_extents+tlrc                                           \
       -expr 'c*a/b' -prefix TSNR.vreg.r01.$subj 

# -----------------------------------------
# warp anat follower datasets (non-linear)

# first perform any pre-warp erode operations
3dmask_tool -input copy_af_FSvent+orig -dilate_input -1                   \
            -prefix copy_af_FSvent_erode
3dmask_tool -input copy_af_FSWe+orig -dilate_input -1                     \
            -prefix copy_af_FSWe_erode

# and apply any warp operations
3dNwarpApply -source copy_af_anat_w_skull+orig                            \
             -master anat_final.$subj+tlrc                                \
             -ainterp wsinc5 -nwarp anatQQ.sub-681_WARP.nii               \
             anatQQ.sub-681.aff12.1D                                      \
             -prefix follow_anat_anat_w_skull
3dNwarpApply -source copy_af_aaseg+orig                                   \
             -master anat_final.$subj+tlrc                                \
             -ainterp NN -nwarp anatQQ.sub-681_WARP.nii                   \
             anatQQ.sub-681.aff12.1D                                      \
             -prefix follow_ROI_aaseg
3dNwarpApply -source copy_af_aeseg+orig                                   \
             -master pb03.$subj.r01.volreg+tlrc                           \
             -ainterp NN -nwarp anatQQ.sub-681_WARP.nii                   \
             anatQQ.sub-681.aff12.1D                                      \
             -prefix follow_ROI_aeseg
3dNwarpApply -source copy_af_FSvent_erode+orig                            \
             -master pb03.$subj.r01.volreg+tlrc                           \
             -ainterp NN -nwarp anatQQ.sub-681_WARP.nii                   \
             anatQQ.sub-681.aff12.1D                                      \
             -prefix follow_ROI_FSvent
3dNwarpApply -source copy_af_FSWe_erode+orig                              \
             -master pb03.$subj.r01.volreg+tlrc                           \
             -ainterp NN -nwarp anatQQ.sub-681_WARP.nii                   \
             anatQQ.sub-681.aff12.1D                                      \
             -prefix follow_ROI_FSWe

# ---------------------------------------------------------
# data check: compute correlations with spherical ~averages
@radial_correlate -nfirst 0 -do_clean yes -rdir radcor.pb03.volreg        \
                  pb03.$subj.r*.volreg+tlrc.HEAD

# ================================== blur ==================================
# blur each volume of each run
foreach run ( $runs )
    3dmerge -1blur_fwhm 4.0 -doall -prefix pb04.$subj.r$run.blur \
            pb03.$subj.r$run.volreg+tlrc
end

# ================================== mask ==================================
# create 'full_mask' dataset (union mask)
foreach run ( $runs )
    3dAutomask -prefix rm.mask_r$run pb04.$subj.r$run.blur+tlrc
end

# create union of inputs, output type is byte
3dmask_tool -inputs rm.mask_r*+tlrc.HEAD -union -prefix full_mask.$subj

# ---- create subject anatomy mask, mask_anat.$subj+tlrc ----
#      (resampled from tlrc anat)
3dresample -master full_mask.$subj+tlrc -input anatQQ.sub-681+tlrc    \
           -prefix rm.resam.anat

# convert to binary anat mask; fill gaps and holes
3dmask_tool -dilate_input 5 -5 -fill_holes -input rm.resam.anat+tlrc  \
            -prefix mask_anat.$subj

# compute tighter EPI mask by intersecting with anat mask
3dmask_tool -input full_mask.$subj+tlrc mask_anat.$subj+tlrc          \
            -inter -prefix mask_epi_anat.$subj

# compute overlaps between anat and EPI masks
3dABoverlap -no_automask full_mask.$subj+tlrc mask_anat.$subj+tlrc    \
            |& tee out.mask_ae_overlap.txt

# note Dice coefficient of masks, as well
3ddot -dodice full_mask.$subj+tlrc mask_anat.$subj+tlrc               \
      |& tee out.mask_ae_dice.txt

# ---- create group anatomy mask, mask_group+tlrc ----
#      (resampled from tlrc base anat, MNI152_2009_template_SSW.nii.gz)
3dresample -master full_mask.$subj+tlrc -prefix ./rm.resam.group      \
           -input                                                     \
           /usr/local/apps/afni/current/linux_centos_7_64/MNI152_2009_template_SSW.nii.gz'[0]'

# convert to binary group mask; fill gaps and holes
3dmask_tool -dilate_input 5 -5 -fill_holes -input rm.resam.group+tlrc \
            -prefix mask_group

# note Dice coefficient of anat and template masks
3ddot -dodice mask_anat.$subj+tlrc mask_group+tlrc                    \
      |& tee out.mask_at_dice.txt

# ================================= scale ==================================
# scale each voxel time series to have a mean of 100
# (be sure no negatives creep in)
# (subject to a range of [0,200])
foreach run ( $runs )
    3dTstat -prefix rm.mean_r$run pb04.$subj.r$run.blur+tlrc
    3dcalc -a pb04.$subj.r$run.blur+tlrc -b rm.mean_r$run+tlrc \
           -c mask_epi_extents+tlrc                            \
           -expr 'c * min(200, a/b*100)*step(a)*step(b)'       \
           -prefix pb05.$subj.r$run.scale
end

# ================================ regress =================================

# compute de-meaned motion parameters (for use in regression)
1d_tool.py -infile dfile_rall.1D -set_nruns 1                             \
           -demean -write motion_demean.1D

# compute motion parameter derivatives (for use in regression)
1d_tool.py -infile dfile_rall.1D -set_nruns 1                             \
           -derivative -demean -write motion_deriv.1D

# convert motion parameters for per-run regression
1d_tool.py -infile motion_demean.1D -set_nruns 1                          \
           -split_into_pad_runs mot_demean

1d_tool.py -infile motion_deriv.1D -set_nruns 1                           \
           -split_into_pad_runs mot_deriv

# create censor file motion_${subj}_censor.1D, for censoring motion 
1d_tool.py -infile dfile_rall.1D -set_nruns 1                             \
    -show_censor_count -censor_prev_TR                                    \
    -censor_motion 0.25 motion_${subj}

# combine multiple censor files
1deval -a motion_${subj}_censor.1D -b outcount_${subj}_censor.1D          \
       -expr "a*b" > censor_${subj}_combined_2.1D

# create bandpass regressors (instead of using 3dBandpass, say)
1dBport -nodata 1286 2.0 -band 0.01 0.1 -invert -nozero > bandpass_rall.1D

# note TRs that were not censored
set ktrs = `1d_tool.py -infile censor_${subj}_combined_2.1D               \
                       -show_trs_uncensored encoded`

# ------------------------------
# create ROI PC ort sets: FSvent

# create a time series dataset to run 3dpc on...

# detrend, so principal components are not affected
foreach run ( $runs )
    # to censor, create per-run censor files
    1d_tool.py -set_run_lengths $tr_counts -select_runs $run              \
               -infile censor_${subj}_combined_2.1D -write rm.censor.r$run.1D

    # do not let censored time points affect detrending
    3dTproject -polort 5 -prefix rm.det_pcin_r$run                        \
               -censor rm.censor.r$run.1D -cenmode KILL                   \
               -input pb03.$subj.r$run.volreg+tlrc

    # make ROI PCs (per run) : FSvent
    3dpc -mask follow_ROI_FSvent+tlrc -pcsave 3                           \
         -prefix rm.ROIPC.FSvent.r${run} rm.det_pcin_r$run+tlrc

    # zero pad censored TRs and further pad to fill across all runs
    1d_tool.py -censor_fill_parent rm.censor.r$run.1D                     \
        -infile rm.ROIPC.FSvent.r${run}_vec.1D                            \
        -write -                                                          \
      | 1d_tool.py -set_run_lengths $tr_counts -pad_into_many_runs $run 1 \
                   -infile - -write ROIPC.FSvent.r$run.1D
end

# ------------------------------
# run the regression analysis
3dDeconvolve -input pb05.$subj.r*.scale+tlrc.HEAD                         \
    -censor censor_${subj}_combined_2.1D                                  \
    -ortvec bandpass_rall.1D bandpass                                     \
    -ortvec ROIPC.FSvent.r01.1D ROIPC.FSvent.r01                          \
    -ortvec mot_demean.r01.1D mot_demean_r01                              \
    -ortvec mot_deriv.r01.1D mot_deriv_r01                                \
    -polort 5                                                             \
    -num_stimts 0                                                         \
    -jobs 32                                                              \
    -fout -tout -x1D X.xmat.1D -xjpeg X.jpg                               \
    -x1D_uncensored X.nocensor.xmat.1D                                    \
    -fitts fitts.$subj                                                    \
    -errts errts.${subj}                                                  \
    -x1D_stop                                                             \
    -cbucket all_betas.$subj                                              \
    -bucket stats.$subj

# -- use 3dTproject to project out regression matrix --
#    (make errts like 3dDeconvolve, but more quickly)
3dTproject -polort 0 -input pb05.$subj.r*.scale+tlrc.HEAD                 \
           -censor censor_${subj}_combined_2.1D -cenmode NTRP             \
           -ort X.nocensor.xmat.1D -prefix errts.${subj}.tproject



# if 3dDeconvolve fails, terminate the script
if ( $status != 0 ) then
    echo '---------------------------------------'
    echo '** 3dDeconvolve error, failing...'
    echo '   (consider the file 3dDeconvolve.err)'
    exit
endif


# display any large pairwise correlations from the X-matrix
1d_tool.py -show_cormat_warnings -infile X.xmat.1D |& tee out.cormat_warn.txt

# display degrees of freedom info from X-matrix
1d_tool.py -show_df_info -infile X.xmat.1D |& tee out.df_info.txt

# create an all_runs dataset to match the fitts, errts, etc.
3dTcat -prefix all_runs.$subj pb05.$subj.r*.scale+tlrc.HEAD

# --------------------------------------------------
# generate fast ANATICOR result: errts.$subj.fanaticor+tlrc

# --------------------------------------------------
# fast ANATICOR: generate local FSWe time series averages
# create catenated volreg dataset
3dTcat -prefix rm.all_runs.volreg pb03.$subj.r*.volreg+tlrc.HEAD

# mask white matter before blurring
3dcalc -a rm.all_runs.volreg+tlrc -b follow_ROI_FSWe+tlrc                 \
       -expr "a*bool(b)" -datum float -prefix rm.all_runs.volreg.mask

# generate ANATICOR voxelwise regressors
# via full Gaussian blur (radius = 30 mm)
3dmerge -1blur_fwhm 60 -doall -prefix Local_FSWe_rall                     \
        rm.all_runs.volreg.mask+tlrc

# QC: similarly blur the mask to get an idea of the coverage
#     (use a float version of the mask for blurring)
3dcalc -a follow_ROI_FSWe+tlrc -expr a -datum float                       \
       -prefix rm.mask.anaticor.float
3dmerge -1blur_fwhm 60 -doall -prefix fanaticor_mask_coverage             \
        rm.mask.anaticor.float+tlrc

# -- use 3dTproject to project out regression matrix --
#    (make errts like 3dDeconvolve, but more quickly)
3dTproject -polort 0 -input pb05.$subj.r*.scale+tlrc.HEAD                 \
           -censor censor_${subj}_combined_2.1D -cenmode NTRP             \
           -dsort Local_FSWe_rall+tlrc                                    \
           -ort X.nocensor.xmat.1D -prefix errts.$subj.fanaticor

# --------------------------------------------------
# create a temporal signal to noise ratio dataset 
#    signal: if 'scale' block, mean should be 100
#    noise : compute standard deviation of errts
3dTstat -mean -prefix rm.signal.all all_runs.$subj+tlrc"[$ktrs]"
3dTstat -stdev -prefix rm.noise.all errts.$subj.fanaticor+tlrc"[$ktrs]"
3dcalc -a rm.signal.all+tlrc                                              \
       -b rm.noise.all+tlrc                                               \
       -c mask_epi_anat.$subj+tlrc                                        \
       -expr 'c*a/b' -prefix TSNR.$subj 

# ---------------------------------------------------
# compute and store GCOR (global correlation average)
# (sum of squares of global mean of unit errts)
3dTnorm -norm2 -prefix rm.errts.unit errts.$subj.fanaticor+tlrc
3dmaskave -quiet -mask full_mask.$subj+tlrc rm.errts.unit+tlrc            \
          > mean.errts.unit.1D
3dTstat -sos -prefix - mean.errts.unit.1D\' > out.gcor.1D
echo "-- GCOR = `cat out.gcor.1D`"

# ---------------------------------------------------
# compute correlation volume
# (per voxel: correlation with masked brain average)
3dmaskave -quiet -mask full_mask.$subj+tlrc errts.$subj.fanaticor+tlrc    \
          > mean.errts.1D
3dTcorr1D -prefix corr_brain errts.$subj.fanaticor+tlrc mean.errts.1D

# compute 2 requested correlation volume(s)
# create correlation volume corr_af_aeseg
3dcalc -a follow_ROI_aeseg+tlrc -b full_mask.$subj+tlrc -expr 'a*b'       \
       -prefix rm.fm.aeseg
3dmaskave -q -mask rm.fm.aeseg+tlrc errts.$subj.fanaticor+tlrc            \
          > mean.ROI.aeseg.1D
3dTcorr1D -prefix corr_af_aeseg errts.$subj.fanaticor+tlrc mean.ROI.aeseg.1D

# create correlation volume corr_af_FSvent
3dcalc -a follow_ROI_FSvent+tlrc -b full_mask.$subj+tlrc -expr 'a*b'      \
       -prefix rm.fm.FSvent
3dmaskave -q -mask rm.fm.FSvent+tlrc errts.$subj.fanaticor+tlrc           \
          > mean.ROI.FSvent.1D
3dTcorr1D -prefix corr_af_FSvent errts.$subj.fanaticor+tlrc mean.ROI.FSvent.1D

# --------------------------------------------------
# compute sum of baseline (all) regressors
3dTstat -sum -prefix sum_baseline.1D X.nocensor.xmat.1D

# ============================ blur estimation =============================
# compute blur estimates
touch blur_est.$subj.1D   # start with empty file

# create directory for ACF curve files
mkdir files_ACF

# -- estimate blur for each run in epits --
touch blur.epits.1D

# restrict to uncensored TRs, per run
foreach run ( $runs )
    set trs = `1d_tool.py -infile X.xmat.1D -show_trs_uncensored encoded  \
                          -show_trs_run $run`
    if ( $trs == "" ) continue
    3dFWHMx -detrend -mask mask_epi_anat.$subj+tlrc                       \
            -ACF files_ACF/out.3dFWHMx.ACF.epits.r$run.1D                 \
            all_runs.$subj+tlrc"[$trs]" >> blur.epits.1D
end

# compute average FWHM blur (from every other row) and append
set blurs = ( `3dTstat -mean -prefix - blur.epits.1D'{0..$(2)}'\'` )
echo average epits FWHM blurs: $blurs
echo "$blurs   # epits FWHM blur estimates" >> blur_est.$subj.1D

# compute average ACF blur (from every other row) and append
set blurs = ( `3dTstat -mean -prefix - blur.epits.1D'{1..$(2)}'\'` )
echo average epits ACF blurs: $blurs
echo "$blurs   # epits ACF blur estimates" >> blur_est.$subj.1D

# -- estimate blur for each run in errts --
touch blur.errts.1D

# restrict to uncensored TRs, per run
foreach run ( $runs )
    set trs = `1d_tool.py -infile X.xmat.1D -show_trs_uncensored encoded  \
                          -show_trs_run $run`
    if ( $trs == "" ) continue
    3dFWHMx -detrend -mask mask_epi_anat.$subj+tlrc                       \
            -ACF files_ACF/out.3dFWHMx.ACF.errts.r$run.1D                 \
            errts.$subj.fanaticor+tlrc"[$trs]" >> blur.errts.1D
end

# compute average FWHM blur (from every other row) and append
set blurs = ( `3dTstat -mean -prefix - blur.errts.1D'{0..$(2)}'\'` )
echo average errts FWHM blurs: $blurs
echo "$blurs   # errts FWHM blur estimates" >> blur_est.$subj.1D

# compute average ACF blur (from every other row) and append
set blurs = ( `3dTstat -mean -prefix - blur.errts.1D'{1..$(2)}'\'` )
echo average errts ACF blurs: $blurs
echo "$blurs   # errts ACF blur estimates" >> blur_est.$subj.1D


# ================== auto block: generate review scripts ===================

# generate a review script for the unprocessed EPI data
gen_epi_review.py -script @epi_review.$subj              \
    -dsets pb00.$subj.r*.tcat+orig.HEAD

# generate scripts to review single subject results
# (try with defaults, but do not allow bad exit status)
gen_ss_review_scripts.py -mot_limit 0.25 -out_limit 0.05 \
    -errts_dset errts.$subj.fanaticor+tlrc.HEAD -exit0   \
    -ss_review_dset out.ss_review.$subj.txt              \
    -write_uvars_json out.ss_review_uvars.json

# ========================== auto block: finalize ==========================

# remove temporary files
\rm -f rm.*

# if the basic subject review script is here, run it
# (want this to be the last text output)
if ( -e @ss_review_basic ) then
    ./@ss_review_basic |& tee out.ss_review.$subj.txt

    # generate html ss review pages
    # (akin to static images from running @ss_review_driver)
    apqc_make_tcsh.py -review_style pythonic -subj_dir . \
        -uvar_json out.ss_review_uvars.json
    tcsh @ss_review_html |& tee out.review_html
    apqc_make_html.py -qc_dir QC_$subj

    echo "\nconsider running: \n\n    afni_open -b /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//PrcsData/sub-681/D02_Preproc_fMRI/QC_$subj/index.html\n"
endif

# return to parent directory (just in case...)
cd ..

echo "execution finished: `date`"




# ==========================================================================
# script generated by the command:
#
# afni_proc.py -subj_id sub-681 -blocks despike tshift align tlrc volreg blur                                             \
#     mask scale regress -radial_correlate_blocks tcat volreg -copy_anat                                                  \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatSS.sub-681.nii                         \
#     -anat_has_skull no -anat_follower anat_w_skull anat                                                                 \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatUAC.sub-681.nii                        \
#     -anat_follower_ROI aaseg anat                                                                                       \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//Freesurfer//sub-681/SUMA/aparc.a2009s+aseg.nii                            \
#     -anat_follower_ROI aeseg epi                                                                                        \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//Freesurfer//sub-681/SUMA/aparc.a2009s+aseg.nii                            \
#     -anat_follower_ROI FSvent epi                                                                                       \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//Freesurfer//sub-681/SUMA/fs_ap_latvent.nii.gz                             \
#     -anat_follower_ROI FSWe epi                                                                                         \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//Freesurfer//sub-681/SUMA/fs_ap_wm.nii.gz                                  \
#     -anat_follower_erode FSvent FSWe -tcat_remove_first_trs 5 -dsets                                                    \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D00_OriginalData/sub-681_ses-sleep-task-spatatt+orig.HEAD \
#     -align_opts_aea -cost lpc+ZZ -giant_move -check_flip -tlrc_base                                                     \
#     MNI152_2009_template_SSW.nii.gz -tlrc_NL_warp -tlrc_NL_warped_dsets                                                 \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681.nii                         \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681.aff12.1D                    \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01/PrcsData/sub-681/D01_Anatomical/anatQQ.sub-681_WARP.nii                    \
#     -tshift_opts_ts -tpattern seq-z -volreg_align_to first                                                              \
#     -volreg_align_e2a -volreg_tlrc_warp -volreg_warp_dxyz 3 -blur_size 4.0                                              \
#     -mask_epi_anat yes -regress_opts_3dD -jobs 32 -regress_motion_per_run                                               \
#     -regress_ROI_PC FSvent 3 -regress_ROI_PC_per_run FSvent                                                             \
#     -regress_make_corr_vols aeseg FSvent -regress_anaticor_fast                                                         \
#     -regress_anaticor_label FSWe -regress_censor_motion 0.25                                                            \
#     -regress_censor_outliers 0.05 -regress_apply_mot_types demean deriv                                                 \
#     -regress_est_blur_epits -regress_est_blur_errts -regress_bandpass 0.01                                              \
#     0.1 -regress_polort 5 -regress_run_clustsim no -html_review_style                                                   \
#     pythonic -out_dir                                                                                                   \
#     /data/SFIM_Vigilance/PRJ_Vigilance_Smk01//PrcsData/sub-681/D02_Preproc_fMRI                                         \
#     -script SC02_Preproc_fMRI.sub-681.sh -volreg_compute_tsnr yes                                                       \
#     -regress_compute_tsnr yes -regress_make_cbucket yes -scr_overwrite

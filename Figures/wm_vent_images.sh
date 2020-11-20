# 11/17/2020 - Isabel Fernandez

# This script creates images of the white matter and ventrical masks created by Freesurfer over anatomical and epi

modlue load afni

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
ORIG_DATA_DIR='/data/SFIM_Vigilance/Data/DSET02/'

# List of subjects
# ----------------
subjects=(`ls ${ORIG_DATA_DIR} | tr -s '\n' ' '`)
subjects=("${subjects[@]/'README'}") # The subject directory contains a README file. This is not a subject ID.
subjects=("${subjects[@]/'dataset_description.json'}") # The subject directory contains a json file. This is not a subject ID.
subjects=("${subjects[@]/'sub-S21'}") # This subject had bad motion and will not be used.

# Generate Images using AFNI
# --------------------------
for SBJ in ${subjects[@]}
do
    cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI
    # Copies anatomical image cropped if needed
    if [ ! -d anat_final.${SBJ}.cropped+tlrc.BRIK ]; then
        3dAutobox -prefix anat_final.${SBJ}.cropped -input anat_final.${SBJ}+tlrc
    fi
    # Copies epi image cropped if needed
    if [ ! -d final_epi_vr_base.cropped+tlrc.BRIK ]; then
        3dAutobox -prefix final_epi_vr_base.cropped -input final_epi_vr_base+tlrc
    fi
    # AFNI Driver picks underlay and overlay acordingly and montages the axial view
    # Images are saved a JPEG in subjects directory
    afni -com "OPEN_WINDOW A.axialimage mont=7x1:10" \
         -com "SWITCH_UNDERLAY A.anat_final.${SBJ}.cropped" \
         -com "SWITCH_OVERLAY A.follow_ROI_FSWe" \
         -com "SET_DICOM_XYZ A -1.0 17.0 17.0 J" \
         -com "SAVE_JPEG A.axialimage ${SBJ}_wm_anat.jpg" \
         -com "OPEN_WINDOW B.axialimage mont=7x1:4" \
         -com "SWITCH_UNDERLAY B.final_epi_vr_base.cropped" \
         -com "SWITCH_OVERLAY B.follow_ROI_FSWe" \
         -com "SET_DICOM_XYZ B -1.0 16.0 22.0 J" \
         -com "SAVE_JPEG B.axialimage ${SBJ}_wm_epi.jpg" \
         -com "OPEN_WINDOW C.axialimage mont=5x1:5" \
         -com "SWITCH_UNDERLAY C.anat_final.${SBJ}.cropped" \
         -com "SWITCH_OVERLAY C.follow_ROI_FSvent" \
         -com "SET_DICOM_XYZ C -1.0 17.0 16.0 J" \
         -com "SAVE_JPEG C.axialimage ${SBJ}_vent_anat.jpg" \
         -com "OPEN_WINDOW D.axialimage mont=5x1:2" \
         -com "SWITCH_UNDERLAY D.final_epi_vr_base.cropped" \
         -com "SWITCH_OVERLAY D.follow_ROI_FSvent" \
         -com "SET_DICOM_XYZ D -1.0 16.0 13.0 J" \
         -com "SAVE_JPEG D.axialimage ${SBJ}_vent_epi.jpg" \
         -com "QUIT" \
     anat_final.${SBJ}.cropped+tlrc final_epi_vr_base.cropped+tlrc follow_ROI_FSWe+tlrc follow_ROI_FSvent+tlrc
done
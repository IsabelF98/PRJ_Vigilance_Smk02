# 11/17/2020 - Isabel Fernandez

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
        3dAutobox -prefix anat_final.${SBJ}.cropped -input anat_final.${SBJ}+tlrc # Copies anatomical image cropped
    fi
    # AFNI Driver picks underlay and overlay acordingly and montages the axial view
    # Images are saved a JPEG in subjects directory
    afni -com "OPEN_WINDOW A.axialimage mont=7x1:10 opacity=4" \
         -com "SWITCH_UNDERLAY A.anat_final.${SBJ}.cropped" \
         -com "SWITCH_OVERLAY A.TSNR.${SBJ}" \
         -com "SET_DICOM_XYZ A -1.0 17.0 17.0 J" \
         -com "SET_PBAR_ALL A.+99 550.0 Spectrum:red_to_blue" \
         -com "SAVE_JPEG A.axialimage ${SBJ}_TSNR_orig_550.jpg" \
         -com "OPEN_WINDOW B.axialimage mont=7x1:10" \
         -com "SWITCH_UNDERLAY B.anat_final.${SBJ}.cropped" \
         -com "SWITCH_OVERLAY B.TSNR.vreg.r01.${SBJ}" \
         -com "SET_DICOM_XYZ B -1.0 17.0 17.0 J" \
         -com "SET_PBAR_ALL B.+99 150.0 Spectrum:red_to_blue" \
         -com "SAVE_JPEG B.axialimage ${SBJ}_TSNR_vreg_150.jpg" \
         -com "QUIT" \
         anat_final.${SBJ}.cropped+tlrc TSNR.${SBJ}+tlrc TSNR.vreg.r01.${SBJ}+tlrc
done

#-com "OPEN_WINDOW B.axialimage mont=7x1:10" \
#         -com "SWITCH_UNDERLAY B.anat_final.${SBJ}.cropped" \
#         -com "SWITCH_OVERLAY B.TSNR.vreg.r01.${SBJ}" \
#         -com "SET_DICOM_XYZ B -1.0 17.0 17.0 J" \
#         -com "SET_PBAR_ALL B.+99 150.0 Spectrum:red_to_blue" \
#         -com "SAVE_JPEG B.axialimage ${SBJ}_TSNR_vreg_150.jpg" \
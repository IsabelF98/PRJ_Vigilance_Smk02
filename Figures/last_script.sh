# 11/19/2020 - Isabel Fernandez

# This script moves all the image files to Figures and deletes the cropped anatomical and epi images
# This script must be run after wm_vent_images.sh and tsnr_images.sh

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'

mv ${PRJDIR}/PrcsData/sub-S??/D02_Preproc_fMRI/sub-S??_vent_*.jpg ./CSF/ # Moving ventrical files images subject directory to Figures/CSF
mv ${PRJDIR}/PrcsData/sub-S??/D02_Preproc_fMRI/sub-S??_wm_*.jpg ./WhiteMatter/ # Moving white matter images from subject directory to Figures/WhiteMatter
mv ${PRJDIR}/PrcsData/sub-S??/D02_Preproc_fMRI/sub-S??_TSNR_*.jpg ./TSNR/ # Moving TSNR images form subject direcoty to Figures/TSNR
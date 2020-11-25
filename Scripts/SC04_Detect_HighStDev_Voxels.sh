# 11/23/2020 - Isabel Fernandez
#
# This script creates a FB mask common to all subjects in DSET02. This mask will used in later
# stages of the analysis to only select ROIs with valid voxels in all subjects

set -e
PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
STD_THRESHOLD=2
cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI
echo "++ INFO: working dir: `pwd`"
for WL in 060 046 030
do
  # Compute mean and standard deviation across time voxel-wise  
  3dTstat -overwrite -mean -sigma -prefix errts.${SBJ}.wl${WL}s.fanaticor.Tsigma errts.${SBJ}.wl${WL}s.fanaticor+tlrc

  # Compute intra-cranial mask with voxels that have low stdev (i.e., this takes care of large vasculature and ventricles
  3dcalc -overwrite \
         -a full_mask.${SBJ}+tlrc \
         -b errts.${SBJ}.wl${WL}s.fanaticor.Tsigma+tlrc[1] \
         -expr "a*isnegative(b-${STD_THRESHOLD})" \
         -prefix full_mask.lowSigma.${SBJ}.wl${WL}s
done

3dcalc -overwrite \
       -a full_mask.lowSigma.${SBJ}.wl060s+tlrc \
       -b full_mask.lowSigma.${SBJ}.wl046s+tlrc \
       -c full_mask.lowSigma.${SBJ}.wl030s+tlrc \
       -expr 'equals(a+b+c,3)' \
       -prefix full_mask.lowSigma.${SBJ}

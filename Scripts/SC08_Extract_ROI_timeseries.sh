set -e
PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk01/'

cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI

OMP_NUM_THREADS=32
NROIsID=`printf %04d ${NROIs}`
AtlasID=`echo Craddock_T2Level_${NROIsID}`

AtlasFile=`echo ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/${SBJ}.${AtlasID}.lowSigma+tlrc`
DataFile=`echo ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/errts.${SBJ}.wl${WL}s.fanaticor+tlrc`
numROIs=`3dinfo -dmax ${AtlasFile}[0]`
echo "** Atlas File = ${AtlasFile}"
echo "** Data  File = ${DataFile}" 
echo "** numROIs    = ${numROIs}"  
# Perform SVD in masks
# ========================== 
for roi in  $(seq 1 1 ${numROIs})
do
   roiID=`printf %03d ${roi}`
   echo "## INFO: ROI[${roiID}]"
   3dmaskSVD -vnorm -mask ${AtlasFile}"[${roi}]" ${DataFile} > errts.${SBJ}.wl${WL}s.fanaticor.lowSigma.${roiID}.WL${WL}.1D
   mv errts.${SBJ}.wl${WL}s.fanaticor.lowSigma.${roiID}.WL${WL}.1D DXX_NROIS${NROIsID}
done
paste ./DXX_NROIS${NROIsID}/errts.${SBJ}.wl${WL}s.fanaticor.lowSigma.???.WL${WL}.1D > errts.${SBJ}.${AtlasID}.wl${WL}s.fanaticor_ts.1D

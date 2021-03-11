set -e
PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
echo ${PRJDIR}

cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI

OMP_NUM_THREADS=32
NROIsID=22
AtlasID=Tagliazucchi_2012

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
   mv errts.${SBJ}.wl${WL}s.fanaticor.lowSigma.${roiID}.WL${WL}.1D DXX_Tagliazucchi_2012
done
paste ./DXX_Tagliazucchi_2012/errts.${SBJ}.wl${WL}s.fanaticor.lowSigma.???.WL${WL}.1D > errts.${SBJ}.${AtlasID}.wl${WL}s.fanaticor_ts.1D

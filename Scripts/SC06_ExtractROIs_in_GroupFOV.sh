# 11/30/2020 - Isabel Fernandez

set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
ROIMinSize=10
OrigAtlasSpace=MNI
NROIS=200

NROIsID=`printf %04d ${NROIS}`
AtlasID=`echo Craddock_T2Level_${NROIsID}`
AtlasDir=`echo ${PRJDIR}/Atlases`
MNIOrigAtlasFile=`echo ${AtlasDir}/${AtlasID}/${AtlasID}.${OrigAtlasSpace}`
MNITempAtlasFile=`echo ${AtlasID}.${OrigAtlasSpace}.TEMP`
MNIFinaAtlasFile=`echo all.${AtlasID}.${OrigAtlasSpace}`
CommonFBrainMask=` echo  ${PRJDIR}/PrcsData/all/full_mask.lowSigma.all`

echo "** AtlasDIR              : ${AtlasDir}"
echo "** MNI Orig Atlas File   : ${MNIOrigAtlasFile}"
echo "** MNI Temp Atlas File   : ${MNITempAtlasFile}"
echo "** MNI Fina Atlas File   : ${MNIFinaAtlasFile}"
echo "** Common FBrain Mask    : ${CommonFBrainMask}"
echo "** ROI Min Size (voxels) : ${ROIMinSize}"

# Create Atlas that only contains ROIs within common mask
# =======================================================
cd ${PRJDIR}/PrcsData/all

# (0) Remove Files prior to start (if not, number of ROIs may decrease with each successive execution of this script)
if [ -f all.${AtlasID}.${OrigAtlasSpace}+tlrc.HEAD  ]; then rm all.${AtlasID}.${OrigAtlasSpace}+tlrc.*; fi
if [ -f all.${AtlasID}.${OrigAtlasSpace}.txt        ]; then rm all.${AtlasID}.${OrigAtlasSpace}.txt; fi
if [ -f all.${AtlasID}.${OrigAtlasSpace}.rankmap.1D ]; then rm all.${AtlasID}.${OrigAtlasSpace}.rankmap.1D; fi

# (1) Bring Craddock Atlas to the same space as the group-level mask
3dresample -overwrite \
           -rmode NN  \
           -inset   ${MNIOrigAtlasFile}+tlrc \
           -master  ${CommonFBrainMask}+tlrc \
           -prefix  ${MNITempAtlasFile} 

# (2) Restrict this temp version of the atlas to only voxels inside the group-level mask
3dcalc -overwrite \
       -a ${MNITempAtlasFile}+tlrc \
       -b ${CommonFBrainMask}+tlrc \
       -expr 'a*b' \
       -prefix ${MNITempAtlasFile}

# (3) Create a new Atlas with only the ROIs that contain at least ${ROIMinSize} voxels
#     in the field of view of all subjects
# ------------------------------------------------------------------------------------
3dresample -overwrite \
           -rmode NN  \
           -inset ${MNIOrigAtlasFile}+tlrc \
           -master ${CommonFBrainMask}+tlrc \
           -prefix ${MNIFinaAtlasFile}

numROIsInOrig=`3dinfo -dmax ${MNITempAtlasFile}+tlrc`
valid=1

for (( i=1; i<=${numROIsInOrig}; i++ ))
do
 validID=`printf %03d ${valid}`
 iID=`printf %03d ${i}`
 3dcalc -overwrite -a ${MNITempAtlasFile}+tlrc -expr "equals(a,${i})" -overwrite -prefix ${MNITempAtlasFile}.${validID}
 numVoxels=`3dROIstats -nomeanout -quiet -nzvoxels -mask ${MNITempAtlasFile}.${validID}+tlrc ${MNITempAtlasFile}.${validID}+tlrc`
 if [ -z "$numVoxels" ]; then numVoxels=0; fi
 echo "** I=${i} | IID=${iID} | VALID=${valid} | VALIDID=${validID} --> Num Voxels = ${numVoxels}"
 if [ ${numVoxels} -ge ${ROIMinSize} ]; then
    valid=$(($valid+1))
    echo "ROI${iID} --> ROI${validID} | Size = ${numVoxels}"
    echo "ROI${iID} --> ROI${validID} | Size = ${numVoxels}" >> ${MNIFinaAtlasFile}.txt
 else
    echo "ROI${iID} DELETED | Size = ${numVoxels}"
    echo "ROI${iID} DELETED | Size = ${numVoxels}" >> ${MNIFinaAtlasFile}.txt
    3dcalc -a ${MNIFinaAtlasFile}+tlrc -expr "a-(${i}*equals(a,${i}))" -prefix ${MNIFinaAtlasFile} -overwrite
 fi
done
3dmerge -overwrite -1rank                \
        -prefix ${MNIFinaAtlasFile}      \
                ${MNIFinaAtlasFile}+tlrc

numROIsFinal=`3dinfo -dmax ${MNIFinaAtlasFile}+tlrc`
rm ${MNITempAtlasFile}.???+tlrc.????
rm ${MNITempAtlasFile}+tlrc.????

echo "=========================================================="
echo "               SUMMARY"
echo "=========================================================="
echo "ORIG NUMBER OF ROIs: ${numROIsInOrig}"
echo "FINAL MASK FILE IS : ${MNIFinaAtlasFile}+tlrc"
echo "ORIGINAL NUM ROIS  : ${NROIS} (${AtlasID}.${OrigAtlasSpace}+tlrc)"
echo "FINAL  NUM ROIS    : ${numROIsFinal} (${MNIFinaAtlasFile}+tlrc)"
echo "=========================================================="
echo "** AtlasDIR              : ${AtlasDir}"
echo "** MNI Orig Atlas File   : ${MNIOrigAtlasFile}"
echo "** MNI Temp Atlas File   : ${MNITempAtlasFile}"
echo "** MNI Fina Atlas File   : ${MNIFinaAtlasFile}"
echo "** Common FBrain Mask    : ${CommonFBrainMask}"
echo "** ROI Min Size (voxels) : ${ROIMinSize}"


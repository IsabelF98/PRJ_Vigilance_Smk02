set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
NROIS_ID=`printf %04d ${NROIS}` 
ATLASID=`echo Craddock_T2Level_${NROIS_ID}`
cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI

# Here I use the mask for 60s becuase it is the most restrictive (less filtering --> more variance available)
3dcalc -overwrite \
       -a ${PRJDIR}/PrcsData/all/all.${ATLASID}.MNI+tlrc \
       -b ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/full_mask.lowSigma.${SBJ}.wl060s+tlrc \
       -expr 'a*b' \
       -prefix ${SBJ}.${ATLASID}.lowSigma

# Give all the ROIs the lowest possible value
3dmerge -overwrite -1rank \
        -prefix ${SBJ}.${ATLASID}.lowSigma \
                ${SBJ}.${ATLASID}.lowSigma+tlrc

# Create a new dataset that contains the following:
# sub-brick#0: all ROIs
# sub-brick#1: ROI01
# ...
# sub-brick#N: ROIN
numROIs=`3dinfo -dmax ${SBJ}.${ATLASID}.lowSigma+tlrc`
echo "** Number of ROIs   : ${numROIs}"

for (( i=1; i<=${numROIs}; i++ ))
do
 iID=`printf %03d ${i}`
 3dcalc -a ${SBJ}.${ATLASID}.lowSigma+tlrc -expr "equals(a,${i})" -overwrite -prefix ${SBJ}.${ATLASID}.lowSigma.${iID}
done

FILES=`ls ${SBJ}.${ATLASID}.lowSigma.???+tlrc.HEAD | tr -s '\n' ' ' | sed 's/.HEAD//'`
echo ${FILES}
3dbucket -overwrite -prefix ${SBJ}.${ATLASID}.lowSigma ${SBJ}.${ATLASID}.lowSigma+tlrc ${FILES}
rm ${SBJ}.${ATLASID}.lowSigma.???+tlrc.????
echo "++ INFO: Output file: ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/${SBJ}.${ATLASID}.lowSigma+tlrc"

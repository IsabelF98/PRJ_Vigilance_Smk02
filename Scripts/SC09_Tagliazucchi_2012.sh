set -e

PRJDIR='/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/'
NROIS_ID=`printf %04d ${NROIS}` 
ATLASID=`echo Tagliazucchi_2012` 
cd ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI

# Add the personalized subcortical ROIs to the Craddock ATLAS
# Also, we use the 60s mask because it is the most restrictive (less filtering --> more variance available)
# These are the ROIS that are added:
# 49 = R-Thalamus, 10 = L-Thalamus, 52 = R-Pallidum, 13 = L-Pallidum, 51 = R-Putamen, 12 = L-Putamen, 50 = R-Caudate, 11 = L-Caudate,
# 58 = R-Accumbens, 26 = L-Accumbens

# Create initial version with the bilateral thalamus
# ==================================================
3dcalc -overwrite \
       -a ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/follow_ROI_aeseg+tlrc \
       -expr 'equals(a,49) + (2*equals(a,10))' \
       -prefix ${SBJ}.${ATLASID}.lowSigma

# Erode the original Thalamus ROI as it is quite big
# ==================================================
3dmask_tool -overwrite \
           -input ${SBJ}.${ATLASID}.lowSigma+tlrc \
           -dilate_input -1 \
           -prefix ${SBJ}.${ATLASID}.lowSigma.eroded
3dcalc -overwrite \
       -a ${SBJ}.${ATLASID}.lowSigma+tlrc \
       -b ${SBJ}.${ATLASID}.lowSigma.eroded+tlrc \
       -expr 'a*b' \
       -prefix ${SBJ}.${ATLASID}.lowSigma

rm ${SBJ}.${ATLASID}.lowSigma.eroded+tlrc.*

# Add Spheres in all other locations according to Tagliazucchi et al. 2012
# ========================================================================
roi_id=3
for coords in 2,-82,20 30,-90,16 -26,-90,16 2,6,44 8,2,-4 -54,2,-8 62,-22,16 -58,-22,8 -2,-14,48 50,-14,52 -38,-14,52 2,54,-8 2,-62,44 50,-66,28 -46,-70,28 34,46,20 -34,42,20 10,-42,48 58,-38,28 -58,-42,28
do
    xcor=`echo $coords | awk -F ',' '{print $1}'`
    ycor=`echo $coords | awk -F ',' '{print $2}'`
    zcor=`echo $coords | awk -F ',' '{print $3}'`
    echo "++ INFO: Adding sphere @ $xcor | $ycor | $zcor --> ${roi_id}"
    3dcalc -LPI -overwrite \
           -a ${SBJ}.${ATLASID}.lowSigma+tlrc \
           -expr "a+${roi_id}*(step(16.0-(x-${xcor})*(x-${xcor})-(y-${ycor})*(y-${ycor})-(z-${zcor})*(z-${zcor})))" \
           -prefix ${SBJ}.${ATLASID}.lowSigma
    roi_id=$((roi_id+1))
done

# Ensure we remove voxels outside the low sigma mask
# ==================================================
3dcalc -overwrite \
       -a ${SBJ}.${ATLASID}.lowSigma+tlrc \
       -b ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/full_mask.lowSigma.${SBJ}.wl060s+tlrc \
       -expr 'a*b' \
       -prefix ${SBJ}.${ATLASID}.lowSigma


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
 echo "++ 3dcalc -a ${SBJ}.${ATLASID}.lowSigma+tlrc -expr equals(a,${i}) -overwrite -prefix ${SBJ}.${ATLASID}.lowSigma.${iID}"
 3dcalc -a ${SBJ}.${ATLASID}.lowSigma+tlrc -expr "equals(a,${i})" -overwrite -prefix ${SBJ}.${ATLASID}.lowSigma.${iID}
done

FILES=`ls ${SBJ}.${ATLASID}.lowSigma.???+tlrc.HEAD | tr -s '\n' ' ' | sed 's/.HEAD//'`
echo ${FILES}
3dbucket -overwrite -prefix ${SBJ}.${ATLASID}.lowSigma ${SBJ}.${ATLASID}.lowSigma+tlrc ${FILES}
rm ${SBJ}.${ATLASID}.lowSigma.???+tlrc.????
echo "++ INFO: Output file: ${PRJDIR}/PrcsData/${SBJ}/D02_Preproc_fMRI/${SBJ}.${ATLASID}.lowSigma+tlrc"

# 2/18/2021   Isabel Fernandez

# This script calls the python script SC03_peridiogram.py that creates the peridiogram of the 4th ventrical signal extracted in step 2

# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/4thVent

# Activate miniconda
echo "++ Activating miniconda"
. /data/fernandezis/Apps/miniconda38/etc/profile.d/conda.sh

# Activate vigilance environment
echo "++ Activating vigilance environment"
conda activate vigilance

# Unset DISPLAY variable so that we don't get an error about access to XDisplay
echo "++ Unsetting the DISPLAY environment variable"
unset DISPLAY

# Call python program
echo "++ Calling Python program: SC03_peridiogram.py -s ${SBJ} -d ${RUN} -w ${DATADIR}"
python ./SC03_peridiogram.py \
    -s ${SBJ} \
    -d ${RUN} \
    -w ${DATADIR}
set -e
# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/Notebooks

# Activate miniconda
echo "++ Activating miniconda"
. /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh

# Activate vigilance environment
echo "++ Activating rapidtide environment"
conda activate vigilance

# Run the program
./N01_SWC.ToSwarm.py -sbj ${SBJ} -run ${RUN} -wl ${WL}


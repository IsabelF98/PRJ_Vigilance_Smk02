# 11/24/2020 - Isabel Fernandez

#!/bin/bash

set -e

cd /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/${SBJ}/D02_Preproc_fMRI

# 3) Create additional clean datasets with other filterings for SWC analyses
# ==========================================================================
TR=`3dinfo -tr errts.${SBJ}.fanaticor+tlrc | awk '{print $0}'`
tr_counts=(`3dinfo -nt pb05.${SBJ}.r0?.scale+tlrc.HEAD | awk '{print $1}' | tr -s '\n' ' '`)
runs=(`count -digits 2 1 ${#tr_counts[@]}`)

echo "TR: " $TR
echo "Counts per run: " ${tr_counts[@]}
echo "Number of runs: " ${runs[@]}

# Concatinate physiological noise and motion corection files
1dcat ROIPC.FSvent.r*.1D > ROIPC.FSvent.rall.1D
1dcat mot_demean.r*.1D > mot_demean.rall.1D
1dcat mot_deriv.r*.1D > mot_deriv.rall.1D

# 3.1) WL = 60s ==> 30 time-points per window
# -------------------------------------------
# hpass 1/60 = 0.017

# make separate regressors per run, with all in one file
for (( i=0; i<${#runs[@]}; i++ ))
do
    let nt=${tr_counts[i]}
    let run=${runs[i]}
    # Creating and padding regressors for bandpass filtering
    1dBport -nodata $nt $TR -band 0.017 0.18 -invert -nozero > rm.bandpass.wl060s.1D
    1d_tool.py -overwrite -infile rm.bandpass.wl060s.1D -pad_into_many_runs $run ${#runs[@]} \
               -set_run_lengths ${tr_counts[@]} \
               -write rm.bandpass.r${runs[i]}.wl060s.1D
    rm rm.bandpass.wl060s.1D
done

1dcat rm.bandpass.r*.wl060s.1D > bandpass_rall.wl060s.1D
rm rm.bandpass.r*.wl060s.1D


3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD              \
    -censor censor_${SBJ}_combined_2.1D                                    \
    -ortvec bandpass_rall.wl060s.1D bandpass                               \
    -ortvec ROIPC.FSvent.rall.1D ROIPC.FSvent.rall                         \
    -ortvec mot_demean.rall.1D mot_demean_rall                             \
    -ortvec mot_deriv.rall.1D mot_deriv_rall                               \
    -polort 5                                                              \
    -num_stimts 0                                                          \
    -jobs 32                                                               \
    -fout -tout -x1D X.xmat.wl060s.1D -xjpeg X.wl060s.jpg                  \
    -x1D_uncensored X.nocensor.xmat.wl060s.1D                              \
    -fitts fitts.wl060s.${SBJ}                                             \
    -errts errts.wl060s.${SBJ}                                             \
    -x1D_stop                                                              \
    -cbucket all_betas.wl060s.${SBJ}                                       \
    -bucket stats.wl060s.${SBJ}

3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -ort X.nocensor.xmat.wl060s.1D -prefix errts.${SBJ}.wl060s.tproject

# display any large pairwise correlations from the X-matrix
1d_tool.py -show_cormat_warnings -infile X.xmat.wl060s.1D |& tee out.cormat_warn.wl060s.txt

# display degrees of freedom info from X-matrix
1d_tool.py -show_df_info -infile X.xmat.wl060s.1D |& tee out.df_info.wl060s.txt

# -- use 3dTproject to project out regression matrix --
#    (make errts like 3dDeconvolve, but more quickly)
3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -dsort Local_FSWe_rall+tlrc \
           -ort X.nocensor.xmat.wl060s.1D -prefix errts.${SBJ}.wl060s.fanaticor

# 3.2) WL = 46s ==> 23 time-points per window
# -------------------------------------------
# hpass 1/46 = 0.022

# make separate regressors per run, with all in one file
for (( i=0; i<${#runs[@]}; i++ ))
do
    let nt=${tr_counts[i]}
    let run=${runs[i]}
    1dBport -nodata $nt $TR -band 0.022 0.18 -invert -nozero > rm.bandpass.wl046s.1D
    1d_tool.py -overwrite -infile rm.bandpass.wl046s.1D -pad_into_many_runs $run ${#runs[@]} \
               -set_run_lengths ${tr_counts[@]} \
               -write rm.bandpass.r${runs[i]}.wl046s.1D
    rm rm.bandpass.wl046s.1D
done

1dcat rm.bandpass.r*.wl046s.1D > bandpass_rall.wl046s.1D
rm rm.bandpass.r*.wl046s.1D

3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD            \
    -censor censor_${SBJ}_combined_2.1D                                  \
    -ortvec bandpass_rall.wl046s.1D bandpass                             \
    -ortvec ROIPC.FSvent.rall.1D ROIPC.FSvent.rall                       \
    -ortvec mot_demean.rall.1D mot_demean_rall                           \
    -ortvec mot_deriv.rall.1D mot_deriv_rall                             \
    -polort 5                                                            \
    -num_stimts 0                                                        \
    -jobs 32                                                             \
    -fout -tout -x1D X.xmat.wl046s.1D -xjpeg X.wl046s.jpg                \
    -x1D_uncensored X.nocensor.xmat.wl046s.1D                            \
    -fitts fitts.wl046s.${SBJ}                                           \
    -errts errts.wl046s.${SBJ}                                           \
    -x1D_stop                                                            \
    -cbucket all_betas.wl046s.${SBJ}                                     \
    -bucket stats.wl046s.${SBJ}

3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -ort X.nocensor.xmat.wl046s.1D -prefix errts.${SBJ}.wl046s.tproject

# display any large pairwise correlations from the X-matrix
1d_tool.py -show_cormat_warnings -infile X.xmat.wl046s.1D |& tee out.cormat_warn.wl046s.txt

# display degrees of freedom info from X-matrix
1d_tool.py -show_df_info -infile X.xmat.wl046s.1D |& tee out.df_info.wl046s.txt

# -- use 3dTproject to project out regression matrix --
#    (make errts like 3dDeconvolve, but more quickly)
3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -dsort Local_FSWe_rall+tlrc \
           -ort X.nocensor.xmat.wl046s.1D -prefix errts.${SBJ}.wl046s.fanaticor

# 3.3) WL = 30s ==> 15 time-points per window
# -------------------------------------------
# hpass 1/30 = 0.033

# make separate regressors per run, with all in one file
for (( i=0; i<${#runs[@]}; i++ ))
do
    let nt=${tr_counts[i]}
    let run=${runs[i]}
    1dBport -nodata $nt $TR -band 0.033 0.18 -invert -nozero > rm.bandpass.wl030s.1D
    1d_tool.py -overwrite -infile rm.bandpass.wl030s.1D -pad_into_many_runs $run ${#runs[@]} \
               -set_run_lengths ${tr_counts[@]} \
               -write rm.bandpass.r${runs[i]}.wl030s.1D
    rm rm.bandpass.wl030s.1D
done

1dcat rm.bandpass.r*.wl030s.1D > bandpass_rall.wl030s.1D
rm rm.bandpass.r*.wl030s.1D

3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD             \
    -censor censor_${SBJ}_combined_2.1D                                   \
    -ortvec bandpass_rall.wl030s.1D bandpass                              \
    -ortvec ROIPC.FSvent.rall.1D ROIPC.FSvent.rall                        \
    -ortvec mot_demean.rall.1D mot_demean_rall                            \
    -ortvec mot_deriv.rall.1D mot_deriv_rall                              \
    -polort 5                                                             \
    -num_stimts 0                                                         \
    -jobs 32                                                              \
    -fout -tout -x1D X.xmat.wl030s.1D -xjpeg X.wl030s.jpg                 \
    -x1D_uncensored X.nocensor.xmat.wl030s.1D                             \
    -fitts fitts.wl030s.${SBJ}                                            \
    -errts errts.wl030s.${SBJ}                                            \
    -x1D_stop                                                             \
    -cbucket all_betas.wl030s.${SBJ}                                      \
    -bucket stats.wl030s.${SBJ}

3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -ort X.nocensor.xmat.wl030s.1D -prefix errts.${SBJ}.wl030s.tproject

# display any large pairwise correlations from the X-matrix
1d_tool.py -show_cormat_warnings -infile X.xmat.wl030s.1D |& tee out.cormat_warn.wl030s.txt

# display degrees of freedom info from X-matrix
1d_tool.py -show_df_info -infile X.xmat.wl030s.1D |& tee out.df_info.wl030s.txt

# -- use 3dTproject to project out regression matrix --
#    (make errts like 3dDeconvolve, but more quickly)
3dTproject -overwrite -polort 0 -input pb05.${SBJ}.r*.scale+tlrc.HEAD \
           -censor censor_${SBJ}_combined_2.1D -cenmode NTRP \
           -dsort Local_FSWe_rall+tlrc \
           -ort X.nocensor.xmat.wl030s.1D -prefix errts.${SBJ}.wl030s.fanaticor

rm ROIPC.FSvent.rall.1D
rm mot_demean.rall.1D
rm mot_deriv.rall.1D
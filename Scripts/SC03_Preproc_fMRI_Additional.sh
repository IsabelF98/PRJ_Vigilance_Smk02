# TO DO:
# 1. edit script to remove files
# 2. edit scripts to concatinate bandpass and motion regressos (only once)
# 3. change all output file names to .test (IMPORTANT)
# 3. run with concatinated files instead of 6
# 4. compare using diff, 3dcalc (subtract), and afni graph

#!/bin/bash

set -e

cd /data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData/${SBJ}/D02_Preproc_fMRI

# 3) Create additional clean datasets with other filterings for SWC analyses
# ==========================================================================
TR=`3dinfo -tr errts.${SBJ}.fanaticor+tlrc | awk '{print $0}'`
tr_counts=(`3dinfo -nt pb05.${SBJ}.r0?.scale+tlrc.HEAD | awk '{print $1}' | tr -s '\n' ' '`)
runs=(`count -digits 1 1 ${#tr_counts[@]}`)

echo "TR: " $TR
echo "Counts per run: " ${tr_counts[@]}
echo "Number of runs: " ${runs[@]}

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
               -write rm.bandpass.r0$run.wl060s.1D
    rm rm.bandpass.wl060s.1D
    # Creating and padding regressors for physiological noise correction
    1d_tool.py -overwrite -infile ROIPC.FSvent.r0$run.1D -pad_into_many_runs $run ${#runs[@]} \
               -set_run_lengths ${tr_counts[@]} \
               -write rm.ROIPC.FSvent.r0$run.padded.1D
done

1dcat rm.bandpass.r*.wl060s.1D > bandpass_rall.wl060s.1D
rm rm.bandpass.r*.wl060s.1D

1dcat rm.ROIPC.FSvent.r*.padded.1D > ROIPC.FSvent.rall.1D
rm rm.ROIPC.FSvent.r*.padded.1D

3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD              \
    -censor censor_${SBJ}_combined_2.1D                                    \
    -ortvec bandpass_rall.wl060s.1D bandpass                               \
    -ortvec ROIPC.FSvent.rall.1D \
    -ortvec ROIPC.FSvent.r01.1D ROIPC.FSvent.r01                           \
    -ortvec ROIPC.FSvent.r02.1D ROIPC.FSvent.r02                           \
    -ortvec ROIPC.FSvent.r03.1D ROIPC.FSvent.r03                           \
    -ortvec ROIPC.FSvent.r04.1D ROIPC.FSvent.r04                           \
    -ortvec ROIPC.FSvent.r05.1D ROIPC.FSvent.r05                           \
    -ortvec ROIPC.FSvent.r06.1D ROIPC.FSvent.r06                           \
    -ortvec mot_demean.r01.1D mot_demean_r01                               \
    -ortvec mot_demean.r02.1D mot_demean_r02                               \
    -ortvec mot_demean.r03.1D mot_demean_r03                               \
    -ortvec mot_demean.r04.1D mot_demean_r04                               \
    -ortvec mot_demean.r05.1D mot_demean_r05                               \
    -ortvec mot_demean.r06.1D mot_demean_r06                               \
    -ortvec mot_deriv.r01.1D mot_deriv_r01                                 \
    -ortvec mot_deriv.r02.1D mot_deriv_r02                                 \
    -ortvec mot_deriv.r03.1D mot_deriv_r03                                 \
    -ortvec mot_deriv.r04.1D mot_deriv_r04                                 \
    -ortvec mot_deriv.r05.1D mot_deriv_r05                                 \
    -ortvec mot_deriv.r06.1D mot_deriv_r06                                 \
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
    let nt= ${tr_counts[i]}
    let run= ${runs[i]}
    1dBport -nodata $nt $TR -band 0.022 0.18 -invert -nozero > bandpass.wl046s.1D
    1d_tool.py -overwrite -infile bandpass.wl046s.1D -pad_into_many_runs $run ${#runs[@]} \
               --set_run_lengths ${tr_counts[@]} \
               -write bandpass.r0$run.wl046s.1D
done

1dcat bandpass.r*.wl046s.1D > bandpass_rall.wl046s.1D

3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD            \
    -censor censor_${SBJ}_combined_2.1D                                  \
    -ortvec bandpass_rall.wl046s.1D bandpass                             \
    -ortvec ROIPC.FSvent.r01.1D ROIPC.FSvent.r01                         \
    -ortvec ROIPC.FSvent.r02.1D ROIPC.FSvent.r02                         \
    -ortvec ROIPC.FSvent.r03.1D ROIPC.FSvent.r03                         \
    -ortvec ROIPC.FSvent.r04.1D ROIPC.FSvent.r04                         \
    -ortvec ROIPC.FSvent.r05.1D ROIPC.FSvent.r05                         \
    -ortvec ROIPC.FSvent.r06.1D ROIPC.FSvent.r06                         \
    -ortvec mot_demean.r01.1D mot_demean_r01                             \
    -ortvec mot_demean.r02.1D mot_demean_r02                             \
    -ortvec mot_demean.r03.1D mot_demean_r03                             \
    -ortvec mot_demean.r04.1D mot_demean_r04                             \
    -ortvec mot_demean.r05.1D mot_demean_r05                             \
    -ortvec mot_demean.r06.1D mot_demean_r06                             \
    -ortvec mot_deriv.r01.1D mot_deriv_r01                               \
    -ortvec mot_deriv.r02.1D mot_deriv_r02                               \
    -ortvec mot_deriv.r03.1D mot_deriv_r03                               \
    -ortvec mot_deriv.r04.1D mot_deriv_r04                               \
    -ortvec mot_deriv.r05.1D mot_deriv_r05                               \
    -ortvec mot_deriv.r06.1D mot_deriv_r06                               \
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
    1dBport -nodata $nt $TR -band 0.033 0.18 -invert -nozero >! bandpass.wl030s.1D
    1d_tool.py -overwrite -infile bandpass.wl030s.1D -pad_into_many_runs $run ${#runs[@]} \
               -set_run_lengths ${tr_counts[@]} \
               -write bandpass.r0$run.wl030s.1D
done

1dcat bandpass.r*.wl030s.1D > bandpass_rall.wl030s.1D

3dDeconvolve -overwrite -input pb05.${SBJ}.r*.scale+tlrc.HEAD             \
    -censor censor_${SBJ}_combined_2.1D                                   \
    -ortvec bandpass_rall.wl030s.1D bandpass                              \
    -ortvec ROIPC.FSvent.r01.1D ROIPC.FSvent.r01                          \
    -ortvec ROIPC.FSvent.r02.1D ROIPC.FSvent.r02                          \
    -ortvec ROIPC.FSvent.r03.1D ROIPC.FSvent.r03                          \
    -ortvec ROIPC.FSvent.r04.1D ROIPC.FSvent.r04                          \
    -ortvec ROIPC.FSvent.r05.1D ROIPC.FSvent.r05                          \
    -ortvec ROIPC.FSvent.r06.1D ROIPC.FSvent.r06                          \
    -ortvec mot_demean.r01.1D mot_demean_r01                              \
    -ortvec mot_demean.r02.1D mot_demean_r02                              \
    -ortvec mot_demean.r03.1D mot_demean_r03                              \
    -ortvec mot_demean.r04.1D mot_demean_r04                              \
    -ortvec mot_demean.r05.1D mot_demean_r05                              \
    -ortvec mot_demean.r06.1D mot_demean_r06                              \
    -ortvec mot_deriv.r01.1D mot_deriv_r01                                \
    -ortvec mot_deriv.r02.1D mot_deriv_r02                                \
    -ortvec mot_deriv.r03.1D mot_deriv_r03                                \
    -ortvec mot_deriv.r04.1D mot_deriv_r04                                \
    -ortvec mot_deriv.r05.1D mot_deriv_r05                                \
    -ortvec mot_deriv.r06.1D mot_deriv_r06                                \
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
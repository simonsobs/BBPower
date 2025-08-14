#!/bin/bash -l

set -e

## Log file
label="_fiducial_model"  # Anything descriptive 
log="./log_compsep${label}"

## programming environment
export OMP_NUM_THREADS=4

basedir=/global/homes/k/kwolz/bbdev/BBPower/jeffreys-prior  ## YOUR RUNNING DIRECTORY
bbpower_dir=/global/homes/k/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER
cd $basedir

bbpower_config=${basedir}/config_fiducial_model.yml

## For MPI parallelization (only on cluster)
# com1="srun -n 10 -c 1 --cpu_bind=cores python -u \
#     ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config"

## Without MPI parallelization
com1="python -u \
     ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config"

# Patch: We have to repeat the definition of outdir, cells_coadded here.
# TODO: Include in new script "bbplotter_nopipe"
outdir=/pscratch/sd/k/kwolz/jeffreys/chains/all_channels/gaussian_fgs/fiducial_model/0000  # YOUR OUTDIR (SAME AS IN CONFIG)
cells_coadded=/global/homes/k/kwolz/bbdev/BBPower/jeffreys-prior/data/cells_coadded_cov_r0_Alens1_baseline_optimistic.fits # YOUR OUTDIR (SAME AS IN CONFIG)

com2="python -m bbpower BBPlotter \
          --cells_coadded_total='dummy.file' \
          --cells_coadded='${cells_coadded}' \
          --cells_noise='dummy.file' \
          --cells_null='dummy.file' \
          --cells_fiducial='${outdir}/cls_fid.fits' \
          --cells_best_fit='${outdir}/cells_model.fits' \
          --param_chains='${outdir}/emcee.npz' \
          --chisq='${outdir}/chi2.npz' \
          --config='${bbpower_config}' \
          --plots='${outdir}/plots'"


echo "Launching pipeline at $(date)"
echo "Logging to ${log}"
echo ${com1}
eval ${com1} # > ${log} 2>&1
echo ${com2}
eval ${com2} # > ${log} 2>&1
echo "Ending batch script at $(date)"
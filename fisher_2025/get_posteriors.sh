#!/bin/bash -l

set -e

## Log file
label="_moments_10yr_12sats"  # Anything descriptive 
log="./log_compsep${label}"

## programming environment
# Nodes: salloc -N4 -C cpu -q interactive -t 04:00:00
module load soconda
export OMP_NUM_THREADS=4


basedir=/global/homes/k/kwolz/bbdev/BBPower/fisher_2025  ## YOUR RUNNING DIRECTORY
bbpower_dir=/global/homes/k/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER

cd $basedir
bbpower_config=configs/compsep_moments.yaml
#bbpower_config=configs/compsep_fiducial.yaml



echo "Launching pipeline at $(date)"
echo "Logging to ${log}"

# srun -n 10 -c 1 --cpu_bind=cores python -u \
python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config # > ${log} 2>&1
# srun -n 10 -c 1 --cpu_bind=cores python -u \
python -u ${bbpower_dir}/bbpower/plotter_nopipe.py --config $bbpower_config # > ${log} 2>&1

echo "Ending batch script at $(date)"
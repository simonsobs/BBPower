#!/bin/bash -l

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A mp107b
#SBATCH -t 2:0:0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

set -e

## Log file
case="wolzetal2024_original"
label="fiducial_${case}"
log="./log_compsep_${label}"

## programming environment
module load soconda
export OMP_NUM_THREADS=4


basedir=/global/homes/k/kwolz/bbdev/BBPower/fisher_2025  ## YOUR RUNNING DIRECTORY
bbpower_dir=/global/homes/k/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER

cd $basedir
bbpower_config="configs/compsep_${label}.yaml"


echo "Launching pipeline at $(date)"
echo "Logging to ${log}"

srun -n 50 -c 4 --cpu_bind=cores \
python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config #> ${log} 2>&1
srun -n 50 -c 4 --cpu_bind=cores \
python -u ${bbpower_dir}/bbpower/plotter_nopipe.py --config $bbpower_config #> ${log} 2>&1

echo "Ending batch script at $(date)"
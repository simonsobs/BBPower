#!/bin/bash -l
#SBATCH --job-name=compsep
#SBATCH --partition=berg          # or berg, normal, redwood
#SBATCH --nodes=1
#SBATCH --ntasks=10                # total number of MPI tasks 
#SBATCH --output=log/slurm-%j.out 

set -e

## programming environment
export OMP_NUM_THREADS=1
eval "$(micromamba shell hook --shell bash)"
micromamba activate pclcat  ## YOUR BBPOWER micromamba environment

basedir=/mnt/users/kwolz/bbdev/BBPower/jeffreys-prior  ## YOUR RUNNING DIRECTORY
bbpower_dir=/mnt/users/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER
cd $basedir

bbpower_config=${basedir}/config_glamdring.yml

echo "Launching pipeline at $(date)"

mpirun -n 10 \
python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config

mpirun -n 10 \
python -u ${bbpower_dir}/bbpower/plotter_nopipe.py --config $bbpower_config

echo "Ending batch script at $(date)"
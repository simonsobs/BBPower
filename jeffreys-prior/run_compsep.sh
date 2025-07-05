#!/bin/bash -l

set -e

## Log file
label="_fiducial_model"  # Anything descriptive 
log="./log_compsep${label}"

## Software environment
export OMP_NUM_THREADS=4

module use --append /global/common/software/sobs/perlmutter/modulefiles
module load soconda/20250314_0.2.0


basedir=/global/homes/k/kwolz/bbdev/BBPower/jeffreys-prior  ## YOUR RUNNING DIRECTORY
bbpower_dir=/global/homes/k/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER
cd $basedir

bbpower_config=${basedir}/config_fiducial_model.yml

## For MPI parallelization
# com="srun -n 10 -c 1 --cpu_bind=cores python -u \
#     ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config"

## Without MPI parallelization
com="python -u \
     ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config"

echo ${com}
echo "Launching pipeline at $(date)"
echo "Logging to ${log}"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"

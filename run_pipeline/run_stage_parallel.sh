#!/bin/bash

export HDF5_USE_FILE_LOCKING=FALSE

## Utils 
export idir="/input/dir/"
export dir="/output/dir/"
export dirall="/dir/containing/covariance/"
mkdir -p ${dir}

nside=512

export OMP_NUM_THREADS=1

## Config
export cf=/dir/config_file.yml

# Set data seeds
seeds=( $(seq -f "%04g" 0000 0005 ) ) #run for 5 simulations
printf "%s\n" "${seeds[@]}" > "${dir}/seeds.txt" 
export nseeds=${#seeds[@]}

# Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks"

# Run each stage
source activate testenv # This loads custom programming environment 
#
cp $cf $dir
# Uncomment for stage to be run in queue 
srun "--ntasks=${ntasks}" "/global/homes/s/susannaz/BBPower/test_hybrid/prepare_directory_parallel.sh"
#srun "--ntasks=${ntasks}" "/global/homes/s/susannaz/BBPower/test_hybrid/bbpowerspecter_parallel.sh" 
#srun "--ntasks=${ntasks}" "/global/homes/s/susannaz/BBPower/test_hybrid/bbpowersummarizer_parallel.sh"
#srun "--ntasks=${ntasks}" "/global/homes/s/susannaz/BBPower/test_hybrid/bbcompsep_parallel.sh"

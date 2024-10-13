#!/bin/bash

## We only need this when using the queue
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH --mail-user=kwolz@sissa.it
#SBATCH --mail-type=fail
#SBATCH -t 00:45:00
#SBATCH -o run_full_data_pipe_d1s1blop.log

## Tested 100 sims on 4 interactive nodes (wall clock time ~30 min)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_south_nside256.fits"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside256/south/apo-david"
export config="paramfiles/paramfile_SAT.yml"
export datadir="/pscratch/sd/k/kwolz/BBPower/sims/nside256/south/r0_inhom/realistic/d1s1/baseline/optimistic"
export cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside256/south/r0_inhom/gaussian/baseline/optimistic/cells"
export chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside256/south/r0_inhom-data/rfree-model/realistic/d1s1/baseline/optimistic"
################################################################################

export OMP_NUM_THREADS=1

## Create the output directory
mkdir -p "${chainsdir}"

## Set data seeds
seeds=( $(seq -f "%04g" 0000 0099 ) )
printf "%s\n" "${seeds[@]}" > "${chainsdir}/seeds.txt" 
export nseeds=${#seeds[@]}

## Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks to run full data pipeline"

## Calculate data-C_ells and sample posteriors
> "${chainsdir}/out.log"
> "${chainsdir}/err.log"
echo "*** Logging to ${chainsdir}/out.log and ${chainsdir}/err.log ***"
srun "--ntasks=${ntasks}" "bash_parallel/bbpower_data.sh"

## Check if output files are there
for i in "${seeds[@]}"; do
    fname="${chainsdir}/test_out_${i}/cells_coadded.fits"
    if [ ! -f $fname ]; then
        echo "ERROR: ${fname} missing."
    fi
done

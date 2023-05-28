#!/bin/bash
# Tested on 50 computing nodes (walltime ~ 60 min)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_david_nside512.fits"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside512/full/apo-david"
export config="test/config_SO.yml"
export datadir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/realistic/d0s0/baseline/optimistic"
export cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic/cells"
export chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside512/full/r0_inhom-data/rfree-model/realistic/d0s0/baseline/optimistic"
################################################################################

export OMP_NUM_THREADS=1

# Create the output directory
mkdir -p "${chainsdir}"

# Set data seeds
seeds=( $(seq -f "%04g" 0000 0499 ) )
printf "%s\n" "${seeds[@]}" > "${chainsdir}/seeds.txt" 
export nseeds=${#seeds[@]}

# Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks to run full data pipeline"

# Calculate data-C_ells and sample posteriors
conda activate bbpower
> "${chainsdir}/out.log"
> "${chainsdir}/err.log"
echo "*** Logging to ${chainsdir}/out.log and ${chainsdir}/err.log ***"
srun "--ntasks=${ntasks}" "bash_parallel/bbpower_data.sh"
conda deactivate
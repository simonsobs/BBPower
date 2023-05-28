#!/bin/bash
# Tested on 50 computing nodes (walltime ~ 60 min)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_david_nside512.fits"
export config="test/config_SO.yml"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside512/full/apo-david"
export simsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic"
export cellsdir="${simsdir}/cells"
export sims_list="${cellsdir}/sims_list.txt"
################################################################################

export OMP_NUM_THREADS=1

# Create the output directories
mkdir -p "${mcmsdir}"
mkdir -p "${cellsdir}"

# Define data splits (arbitrary seed 0000)
> "${cellsdir}/splits_list.txt"
for val in $splits; do
    echo "${simsdir}/0000/${val}" >> "${cellsdir}/splits_list.txt"
done

# Set simulation seeds
sim_seeds=( $(seq -f "%04g" 0000 0499 ) )
> $sims_list
for i in "${sim_seeds[@]}"; do
    echo "${simsdir}/${i}" >> $sims_list
done

# Compute MCMs and simulation C_ells 
bash "bash_parallel/bbpower_sims.sh"
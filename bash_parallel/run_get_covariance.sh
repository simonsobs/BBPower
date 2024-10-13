#!/bin/bash

## Tested on login node for 100 sims (walltime ~3min)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_david_nside256.fits"
export config="paramfiles/paramfile_SAT_covar.yml"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside256/full/apo-david"
export simsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside256/full/r0_inhom/gaussian/baseline/optimistic"
export cellsdir="${simsdir}/cells"
################################################################################

## Create the output directories
mkdir -p "${cellsdir}"

## Create fiducial spectra
python "examples/generate_SO_spectra_baseline_optimistic.py" $cellsdir 

## Define data splits for covariance sacc (here choose seed 0000)
> "${cellsdir}/splits_list.txt"
for val in $splits; do
    echo "${simsdir}/0000/${val}" >> "${cellsdir}/splits_list.txt"
done

## Make list of sims
> "${cellsdir}/cells_all_sims.txt"
seeds=( $(seq -f "%03g" 000 099 ) )
for seed in "${seeds[@]}"; do
    echo "${cellsdir}/cells_all_splits_sim${seed}.fits" >> "${cellsdir}/cells_all_sims.txt"
done

## Compute covariance matrix
python -m bbpower BBPowerSummarizer \
	--splits_list="${cellsdir}/splits_list.txt" \
	--bandpasses_list="/pscratch/sd/k/kwolz/BBPower/examples/data/bpass_list_delta.txt" \
	--cells_all_splits="${cellsdir}/cells_all_splits_sim000.fits" \
	--cells_all_sims="${cellsdir}/cells_all_sims.txt" \
	--cells_coadded_total="${cellsdir}/cells_coadded_total.fits" \
	--cells_coadded="${cellsdir}/cells_coadded.fits" \
	--cells_noise="${cellsdir}/cells_noise.fits" \
	--cells_null="${cellsdir}/cells_null.fits" \
	--config="${config}"

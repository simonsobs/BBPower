#!/bin/bash

## This script is a patch - it runs BBPowerSpecter once and has the purpose of 
## computing the PCL mode couping matrices. This has do be done just once per 
## mask.

## Tested on login node (wall clock time <20 min).
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_south_nside256.fits"
export config="paramfiles/paramfile_SAT.yml"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside256/south/apo-david"
export simsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside256/south/r0_inhom/gaussian/baseline/optimistic"
export cellsdir="${simsdir}/cells"
export sims_list="${cellsdir}/sims_list.txt"
################################################################################


## Which data seed we choose doesn't matter.
export data_seed=0000

## Creates simsdirs
mkdir -p "${mcmsdir}"
mkdir -p "${cellsdir}"
mkdir -p "${simsdir}/${data_seed}"
echo "sim seed ${data_seed}"

## Defines data splits
splits_list="${simsdir}/${data_seed}/splits_list.txt"
> $splits_list
for val in $splits; do
    echo "${simsdir}/${data_seed}/${val}" >> $splits_list
done

## Writes coadded maps to disk (directories are generated on-the-fly)
python examples/write_obs_maps.py $simsdir $data_seed

## Computes power spectra of data splits
python -m bbpower BBPowerSpecter \
    --splits_list="${splits_list}" \
    --masks_apodized="${mask_apodized}" \
    --bandpasses_list="/pscratch/sd/k/kwolz/BBPower/examples/data/bpass_list_delta.txt" \
    --sims_list="${cellsdir}/sims_list.dum" \
    --beams_list="/pscratch/sd/k/kwolz/BBPower/examples/data/beams_list.txt" \
    --cells_all_splits="${cellsdir}/cells_all_splits_sim${data_seed#0}.fits" \
    --cells_all_sims="${cellsdir}/cells_all_sims.dum" \
    --mcm="${mcmsdir}/mcm.dum" \
    --config="${config}"

## Deletes maps
rm -rf "${simdir}/${data_seed}"

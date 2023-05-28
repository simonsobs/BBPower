#!/bin/bash
# This script, when run on many NERSC nodes, distributes the simulation seeds 
# over the available nodes. Global hyperparameters and other global information 
# is shared via the environment variables. 
# For this specific script, make sure that the environment variables 
# "mask_apodized", "simsdir", "mcmsdir", "cellsdir", "config" and "splits"
# have been defined in a preceding script.
{
# Reads in seeds to sample 
IFS=$'\r\n' GLOBIGNORE='*' command eval  'seeds=($(cat ${cellsdir}/seeds.txt))'

# Runs pipeline 'nseeds' times in parallel.
for k in $( eval echo {0..$( expr $nseeds / $(( $SLURM_NTASKS + 1 )) )} )
do
        tmp1=$(( $SLURM_PROCID + $k*$SLURM_NTASKS ))
        tmp2=$(( $tmp1 < $nseeds ? $tmp1 : $nseeds )) # minimum
        export data_seed=${seeds[$tmp2]}

        # Creates simsdirs
        mkdir -p "${simsdir}/${data_seed}"
        echo "sim seed ${data_seed}"

        # Defines data splits
        splits_list="${simsdir}/${data_seed}/splits_list.txt"
        > $splits_list
        for val in $splits; do
            echo "${simsdir}/${data_seed}/${val}" >> $splits_list
        done
        
        # Writes coadded maps to disk (directories are generated on-the-fly)
        python examples/write_obs_maps.py $simsdir $data_seed

        # Computes power spectra of data splits
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
        
        # Deletes maps
        rm -rf "${simdir}/${data_seed}"
        
        echo "sim seed ${data_seed} done at task ${SLURM_PROCID}"
done
} >> "${cellsdir}/out.log" 2>> "${cellsdir}/err.log"
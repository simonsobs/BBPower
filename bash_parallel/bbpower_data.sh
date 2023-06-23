#!/bin/bash
# This script, when run on many NERSC nodes, distributes the simulation seeds 
# over the available nodes. Global hyperparameters and other global information 
# is shared via the environment variables. 
# For this specific script, make sure that the environment variables 
# "mask_apodized", "datadir", "mcmsdir", "chainsdir", "config" and "splits" 
# have been defined in a preceding script.
{
# Reads in seeds to sample 
IFS=$'\r\n' GLOBIGNORE='*' command eval  'seeds=($(cat ${chainsdir}/seeds.txt))'

# Runs pipeline 'nseeds' times in parallel.
for k in $( eval echo {0..$( expr $nseeds / $(( $SLURM_NTASKS + 1 )) )} )
do
        tmp1=$(( $SLURM_PROCID + $k*$SLURM_NTASKS ))
        tmp2=$(( $tmp1 < $nseeds ? $tmp1 : $nseeds )) # minimum
        export data_seed=${seeds[$tmp2]}

        # Creates data outdirs
        mkdir -p "${chainsdir}/test_out_${data_seed}"
        echo "data seed ${data_seed}"

        # Defines data splits
        splits_list="${chainsdir}/test_out_${data_seed}/splits_list_${data_seed}.txt"
        > $splits_list
        for val in $splits; do
            echo "${datadir}/${data_seed}/${val}" >> $splits_list
        done
        
        # Writes coadded maps to disk (directories are generated on-the-fly)
        python examples/write_obs_maps.py $datadir $data_seed

        # Computes power spectra of data splits
        python -m bbpower BBPowerSpecter \
            --splits_list="${splits_list}" \
            --masks_apodized="${mask_apodized}" \
            --bandpasses_list="/pscratch/sd/k/kwolz/BBPower/examples/data/bpass_list_delta.txt" \
            --sims_list="${chainsdir}/sims_list.dum" \
            --beams_list="/pscratch/sd/k/kwolz/BBPower/examples/data/beams_list.txt" \
            --cells_all_splits="${chainsdir}/test_out_${data_seed}/cells_all_splits.fits" \
            --cells_all_sims="${chainsdir}/cells_all_sims.dum" \
            --mcm="${mcmsdir}/mcm.dum" \
            --config="${config}"
        
        # Deletes maps
        rm -rf "${datadir}/${data_seed}"
        
        # Summarizes power spectra from splits
        python -m bbpower BBPowerSummarizer \
            --splits_list="${splits_list}" \
            --bandpasses_list="/pscratch/sd/k/kwolz/BBPower/examples/data/bpass_list_delta.txt" \
            --cells_all_splits="${chainsdir}/test_out_${data_seed}/cells_all_splits.fits" \
            --cells_all_sims="${chainsdir}/test_out_${data_seed}/cells_all_sims.dum" \
            --cells_coadded_total="${chainsdir}/test_out_${data_seed}/cells_coadded_total.fits" \
            --cells_coadded="${chainsdir}/test_out_${data_seed}/cells_coadded.fits" \
            --cells_noise="${chainsdir}/test_out_${data_seed}/cells_noise.fits" \
            --cells_null="${chainsdir}/test_out_${data_seed}/cells_null.fits" \
            --config="${config}"
    
        # Uses separate Cells and covariance to sample model posterior
        python -m bbpower BBCompSep \
            --cells_coadded="${chainsdir}/test_out_${data_seed}/cells_coadded.fits" \
            --cells_noise="${chainsdir}/test_out_${data_seed}/cells_noise.fits" \
            --cells_fiducial="${cellsdir}/cls_fid.fits" \
            --cells_coadded_cov="${cellsdir}/cells_coadded.fits" \
            --output_dir="${chainsdir}/test_out_${data_seed}" \
            --config_copy="${chainsdir}/test_out_${data_seed}/config_copy.yml" \
            --config="${config}"
        
        echo "data seed ${data_seed} done at task ${SLURM_PROCID}"
done
} >> "${chainsdir}/out.log" 2>> "${chainsdir}/err.log"
#!/bin/bash
# This script, when run on many NERSC nodes, distributes the simulation seeds 
# over the available nodes. Global hyperparameters and other global information 
# is shared via the environment variables. 
# For this specific script, make sure that the environment variables 
# "chainsdir", "config", "splits", "cellsdir"
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
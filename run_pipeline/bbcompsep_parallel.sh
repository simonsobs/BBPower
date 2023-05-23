#!/bin/bash

# Reads in seeds to sample 
IFS=$'\r\n' GLOBIGNORE='*' command eval  'seeds=($(cat ${dir}/seeds.txt))'

# Runs pipeline 'nseeds' times in parallel.
for k in $( eval echo {0..$( expr $nseeds / $(( $SLURM_NTASKS + 1 )) )} )
do
        tmp1=$(( $SLURM_PROCID + $k*$SLURM_NTASKS ))
        tmp2=$(( $tmp1 < $nseeds ? $tmp1 : $nseeds )) # numerical min
        export data_seed=${seeds[$tmp2]}
	
	export simdir=${dir}${data_seed}/
	export output=$simdir
	export config=$output"config.yml"
	cp $dir"config.yml" $output"config.yml"

	sed -i "s|ODIR|${dir}|" $config
	sed -i "s/0X/${data_seed}/" $config
        sed -i "s/0Y/${data_seed}/" $config
	
	# Only run seed if param_chains file is not there
        [ -f ${output}/param_chains.npz ] ||
	    echo "data seed ${data_seed}"
	    
            # Runs BBCompSep
	    python3 -m bbpower BBCompSep \
		    --cells_coadded=${output}cells_coadded.sacc \
		    --cells_noise=${output}cells_noise.sacc \
		    --cells_fiducial=${simdir}cls_fid_residual_masked_${data_seed}.fits \
		    --cells_coadded_cov=${dirall}/cells_coadded.fits \ #this is the precomputed file that contains the covariance 
		    --param_chains=${output}param_chains.npz \
		    --config_copy=${output}config_copy.yml \
		    --config=$config
	    
	    echo "data seed ${data_seed} done at task ${SLURM_PROCID}"
done

#!/bin/bash

# Reads in seeds to sample 
IFS=$'\r\n' GLOBIGNORE='*' command eval  'seeds=($(cat ${mdir}/seeds.txt))'

# Runs pipeline 'nseeds' times in parallel.
for k in $( eval echo {0..$( expr $nseeds / $(( $SLURM_NTASKS + 1 )) )} )
do
        tmp1=$(( $SLURM_PROCID + $k*$SLURM_NTASKS ))
        tmp2=$(( $tmp1 < $nseeds ? $tmp1 : $nseeds )) # numerical min
        export data_seed=${seeds[$tmp2]}

	echo $data_seed

	export simdir=${mdir}${data_seed}/
	export output=$simdir

	export config=$mdir"config_hybrid.yml"

	# Compile txt file with all Cls in 
	rm -rf ${output}cells_all_sims.txt
	echo ${output}"cells_all_splits_sim0.fits">> ${output}cells_all_sims.txt
	for nsim in {1..499}
	do
	    echo ${dirall}"cells_all_splits_sim${nsim}.fits" >> ${output}cells_all_sims.txt
	done

	# Creating full simulations list"
	rm -rf ${output}list_sims.txt
	echo $simdir >> ${output}list_sims.txt
	for nsim in {001..499}
	do
	    sim=/dir/to/sim/0${nsim}/
	    echo $sim >> ${output}list_sims.txt
	done

	# Only run seed if param_chains file is not there
	[ -f ${output}/param_chains.npz ] ||
	    echo "data seed ${data_seed}"
	    
            # Runs BBPowerSpecter
	    python3 -m bbpower BBPowerSpecter \
		    --splits_list=${simdir}splits_list.txt \
		    --masks_apodized=./test_hybrid/masks_SAT_ns512.fits \
		    --bandpasses_list=./examples/data/bpass_list.txt \
		    --sims_list=${output}list_sims.txt \
		    --beams_list=./examples/data/beams_list.txt \
		    --cells_all_splits=${output}cells_all_splits.sacc \
		    --cells_all_sims=${output}cells_all_sims.txt \
		    --mcm=${output}mcm.dum \
		    --config=$config
	    
	echo "data seed ${data_seed} done at task ${SLURM_PROCID}"
done

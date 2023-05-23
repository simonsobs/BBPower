#!/bin/bash

# Reads in seeds to sample 
IFS=$'\r\n' GLOBIGNORE='*' command eval  'seeds=($(cat ${dir}/seeds.txt))'

# Runs pipeline 'nseeds' times in parallel.
for k in $( eval echo {0..$( expr $nseeds / $(( $SLURM_NTASKS + 1 )) )} )
do
        tmp1=$(( $SLURM_PROCID + $k*$SLURM_NTASKS ))
        tmp2=$(( $tmp1 < $nseeds ? $tmp1 : $nseeds )) # numerical min
        export data_seed=${seeds[$tmp2]}

	echo $data_seed

	export simdir=${dir}${data_seed}/
	export output=$simdir
	export covdir=$dirall
	cp $dir"config.yml" $output"config.yml"
	export config=$output"config_hybrid.yml"

	#sed -i -r "s/ZZ/block_diagonal/" $config #Calculate covariance
	sed -i -r "s|ZZ|$covdir|" $config #Use pre-computed covariance
	
        # Runs BBPowerSummarizer
	python3 -m bbpower BBPowerSummarizer \
		--splits_list=${simdir}splits_list.txt \
		--bandpasses_list=./examples/data/bpass_list.txt \
		--cells_all_splits=${output}cells_all_splits.sacc \
		--cells_all_sims=${output}cells_all_sims.txt \
		--cells_coadded_total=${output}cells_coadded_total.sacc \
		--cells_coadded=${output}cells_coadded.sacc \
		--cells_noise=${output}cells_noise.sacc \
		--cells_null=${output}cells_null.sacc \
		--config=${config}
    
	echo "data seed ${data_seed} done at task ${SLURM_PROCID}"
done

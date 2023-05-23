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

	# Creating simulated residual directory if not existent
	mkdir -p $simdir
	cp ${simdir}masked_residualmaps_${data_seed}_split0.fits ${simdir}SO_SAT_obs_map_split_1of4.fits
	cp ${simdir}masked_residualmaps_${data_seed}_split1.fits ${simdir}SO_SAT_obs_map_split_2of4.fits
	cp ${simdir}masked_residualmaps_${data_seed}_split2.fits ${simdir}SO_SAT_obs_map_split_3of4.fits
	cp ${simdir}masked_residualmaps_${data_seed}_split3.fits ${simdir}SO_SAT_obs_map_split_4of4.fits

	# Creating splits list
	rm -rf ${simdir}splits_list.txt
	echo ${simdir}SO_SAT_obs_map_split_1of4.fits >> ${simdir}splits_list.txt
	echo ${simdir}SO_SAT_obs_map_split_2of4.fits >> ${simdir}splits_list.txt
	echo ${simdir}SO_SAT_obs_map_split_3of4.fits >> ${simdir}splits_list.txt
	echo ${simdir}SO_SAT_obs_map_split_4of4.fits >> ${simdir}splits_list.txt
    
	echo "prepared directory ${data_seed} at task ${SLURM_PROCID}"
done

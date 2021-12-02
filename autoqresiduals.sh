#!/bin/bash

mdir="test_hybrid/hybrid_masked_alphap/"
cf=$mdir"config.yml"
for k in {0..3}
do
    for j in 0000 #{0000..0020}
    do
        #output=$mdir"sim0$k/output$j/"
	output=$mdir"sim0$k/output$j/"
        mkdir -p $output

	cp test_BBSims_Hybrid/config_hybrid.yaml ${mdir}config.yml
	#cp test_BBSims_Hybrid/config_hybrid.yaml ${output}config.yml
	
        cp $cf $output
        config=$output"config.yml"
        sed -i "s/sim0X/sim0$k/" $config
        sed -i "s/paramsY/params_$j/" $config

        datadir="/mnt/zfsusers/susanna/BBHybrid/data/sim0$k/"
	#datadir="/mnt/zfsusers/susanna/BBHybrid/data/sim00_noiselessMA/"
        coadd=$datadir"cls_coadd_residual_masked_$j.fits"
        noise=$datadir"cls_noise_residual_masked_$j.fits"
        fid=$datadir"cls_fid_residual_masked_$j.fits"
        chain=$output"param_chains.npz"
        configcopy=$output"config_copy.yml"

	addqueue -s -q berg -c '2 hours' -m 4 -s -n 1x12 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
    done
done

#!/bin/bash

mdir="residual_noiseless/"
cf="residual_noiseless/config.yml"

for k in {0..3}
do
    for j in {0..20}
    do
        output=$mdir"sim0$k/output$j/"
        mkdir -p $output

        cp $cf $output
        config=$output"config.yml"
        sed -i "s/sim0X/sim0$k/" $config
        sed -i "s/paramsY/params$j/" $config

        datadir="/mnt/zfsusers/mabitbol/BBHybrid/data/sim0$k/"
        coadd=$datadir"cls_coadd$j.fits"
        noise=$datadir"cls_noise$j.fits"
        fid=$datadir"cls_fid$j.fits"
        chain=$output"param_chains.npz"
        configcopy=$output"config_copy.yml"

        addqueue -q berg -c "2 days" -m 1 -s -n 1x8 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
    done
done


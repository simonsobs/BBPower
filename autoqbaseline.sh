#!/bin/bash

mdir="baseline_full/"
for k in {0..3}
do
    for j in {0000..0020}
    do
        output=$mdir"sim0$k/output$j/"
        mkdir -p $output

        datadir="/mnt/zfsusers/mabitbol/BBHybrid/data/sim0$k/"
        coadd=$datadir"cls_coadd_baseline_$j.fits"
        noise=$datadir"cls_noise_baseline_$j.fits"
        fid=$datadir"cls_fid_baseline_$j.fits"
        chain=$output"param_chains.npz"
        configcopy=$output"config_copy.yml"
        config=$mdir"config.yml"

        addqueue -q cmb -c "2 hours" -m 1 -s -n 1x8 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
    done
done

mdir="baseline_masked/"
for k in {0..3}
do
    for j in {0000..0020}
    do
        output=$mdir"sim0$k/output$j/"
        mkdir -p $output

        datadir="/mnt/zfsusers/mabitbol/BBHybrid/data/sim0$k/"
        coadd=$datadir"cls_coadd_baseline_masked_$j.fits"
        noise=$datadir"cls_noise_baseline_masked_$j.fits"
        fid=$datadir"cls_fid_baseline_masked_$j.fits"
        chain=$output"param_chains.npz"
        configcopy=$output"config_copy.yml"
        config=$mdir"config.yml"

        addqueue -q cmb -c "2 hours" -m 1 -s -n 1x8 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
    done
done


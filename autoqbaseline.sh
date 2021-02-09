#!/bin/bash

for k in {0..3}
do
    for j in {0..19}
    do
        output="sim0$k/output$j/"
        mkdir -p $output

        datadir="/mnt/zfsusers/mabitbol/BBHybrid/data/sim0$k/"
        coadd=$datadir"cls_coadd_base$j.fits"
        noise=$datadir"cls_noise_base$j.fits"
        fid=$datadir"cls_fid_base$j.fits"
        chain=$output"param_chains.npz"
        configcopy=$output"config_copy.yml"
        config="config.yml"

        addqueue -q cmb -c "2 days" -m 1 -s -n 1x12 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
    done
done

#--cells_coadded=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_coadd_base0.fits   
#--cells_noise=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_noise_base0.fits   
#--cells_fiducial=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_fid_base0.fits   
#--param_chains=./moments/baseline0chi2/output/param_chains.npz   
#--config_copy=./moments/baseline0chi2/output/config_copy.yml   
#--config=./moments/baseline0chi2/config.yml 

#python3 -m bbpower BBCompSep   --cells_coadded=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_coadd_base0.fits   --cells_noise=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_noise_base0.fits   --cells_fiducial=/mnt/zfsusers/mabitbol/BBHybrid/data/sim00/cls_fid_base0.fits   --param_chains=./moments/baseline0chi2/output/param_chains.npz   --config_copy=./moments/baseline0chi2/output/config_copy.yml   --config=./moments/baseline0chi2/config.yml 

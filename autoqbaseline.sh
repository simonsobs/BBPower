#!/bin/bash

##mdir="baseline_full/"
##for k in {0..3}
##do
##    for j in 0000 #{0000..0020}
##    do
##        output=$mdir"sim0$k/output$j/"
##        mkdir -p $output
##	cp config.yml $output
##	cp config.yml $mdir
##
##        datadir="/mnt/zfsusers/susanna/BBHybrid/data/sim0$k/"
##        coadd=$datadir"cls_coadd_baseline_$j.fits"
##        noise=$datadir"cls_noise_baseline_$j.fits"
##        fid=$datadir"cls_fid_baseline_$j.fits"
##        chain=$output"param_chains.npz"
##        configcopy=$output"config_copy.yml"
##        config=$mdir"config.yml"
##	  
##        addqueue -q berg -c "2 hours" -m 1 -s -n 1x8 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
##    done
##done
ellmin=2 #30 #2 #10
#simt=d1s1_maskpysm_Bonly
#simt=d1s1_maskpysm_Bonly_r0.01
#simt=d1s1_maskpysm_Bonly_r0.01_whitenoiONLY
simt=d1s1_maskpysm_Bonly_whitenoiONLY
fsky=0.8 #0.6 #0.8
#mdir="/mnt/extraspace/susanna/BBHybrid/baseline_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}/"
#mdir="/mnt/extraspace/susanna/BBHybrid/baseline_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_r0.01/"
#mdir="/mnt/extraspace/susanna/BBHybrid/baseline_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_r0.01_whitenoiONLY/"
mdir="/mnt/extraspace/susanna/BBHybrid/baseline_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_whitenoiONLY_HandLlik/"
cf=$mdir"config.yml"
for j in {0000..0020} #0000
do
    output=$mdir"output$j/"
    mkdir -p $output

    #cp test_BBSims_Hybrid/config_baseline_d1s1.yaml ${mdir}config.yml
    cp test_BBSims_Hybrid/config_baseline_fsky.yaml ${mdir}config.yml

    cp $cf $output
    config=$output"config.yml"
    sed -i "s/KK/${ellmin}/" $config

    echo $config

    datadir="/mnt/zfsusers/susanna/BBHybrid/data/sim${simt}/fsky${fsky}/"
    coadd=$datadir"cls_coadd_baseline_masked_$j.fits"
    noise=$datadir"cls_noise_baseline_masked_$j.fits"
    fid=$datadir"cls_fid_baseline_masked_$j.fits"
    chain=$output"param_chains.npz"
    configcopy=$output"config_copy.yml"
    
    addqueue -s -q berg -c 'baseline_fsky0.8_ellm2 2.5hr' -m 4 -s -n 1x12 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_coadded_cov=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config

done



#!/bin/bash

ellmin=2 #2 #10 #30
#simt=d1s1_maskpysm_Bonly
#simt=d1s1_maskpysm_Bonly_r0.01
simt=d1s1_maskpysm_Bonly_r0.01_whitenoiONLY
#simt=d1s1_maskpysm_Bonly_whitenoiONLY

fsky=0.8 #0.3 0.6 #0.8 
zz=masked_ #full_ 

#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_planck/realistic/d1s1/fsky${fsky}/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_epsilonfixed/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_r0.01/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_r0.01_whitenoiONLY/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_whitenoiONLY/"
#mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_whitenoiONLY_HandLlik/"
mdir="/mnt/extraspace/susanna/BBHybrid/hybrid_masked_outs/mask_pysm/realistic/d1s1/fsky${fsky}_ellmin${ellmin}_r0.01_whitenoiONLY_HandLlik/"
cf=$mdir"config.yml"

for j in {0000..0020} #0000
do
    output=$mdir"output$j/"
    echo $output
    mkdir -p $output
    
    #cp test_BBSims_Hybrid/config_hybrid.yaml ${mdir}config.yml
    #cp test_BBSims_Hybrid/config_hybrid_fixedepsilon.yaml ${mdir}config.yml
    cp test_BBSims_Hybrid/config_hybrid_ellmin_fsky.yaml ${mdir}config.yml
    
    cp $cf $output
    config=$output"config.yml"
    sed -i "s/simJ/sim${simt}/" $config
    sed -i "s/fskyX/fsky${fsky}/" $config
    sed -i "s/paramsY/params_$j/" $config
    sed -i "s/Zhybrid/${zz}hybrid/" $config

    sed -i "s/KK/${ellmin}/" $config
    
    datadir="/mnt/zfsusers/susanna/BBHybrid/data/sim${simt}/fsky${fsky}/"
    #datadir="/mnt/zfsusers/susanna/BBHybrid/data/simd1s1_maskpysm_Bonly/fsky${fsky}/"
    coadd=$datadir"cls_coadd_residual_masked_$j.fits"
    noise=$datadir"cls_noise_residual_masked_$j.fits"
    fid=$datadir"cls_fid_residual_masked_$j.fits"
    chain=$output"param_chains.npz"
    configcopy=$output"config_copy.yml"
    
    addqueue -s -q cmb -c 'res_fsky0.8_ellm10_h&l 2.5 hour' -m 4 -s -n 1x12 /usr/bin/python3 -m bbpower BBCompSep   --cells_coadded=$coadd   --cells_coadded_cov=$coadd   --cells_noise=$noise   --cells_fiducial=$fid   --param_chains=$chain   --config_copy=$configcopy   --config=$config
done


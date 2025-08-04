#!/bin/bash -l

data_dir=/mnt/extraspace/kwolz/jeffreys-prior/data  ## YOUR DATA DIRECTORY
mkdir -p $data_dir
cp camb_lens_nobb.dat $data_dir
cp camb_lens_r1.dat $data_dir
cd $data_dir

wget https://portal.nersc.gov/cfs/sobs/users/so_bb/cells_coadded_all_channels_gaussian_fgs.tar.gz
tar -xzvf cells_coadded_all_channels_gaussian_fgs.tar.gz -C $data_dir

wget https://portal.nersc.gov/cfs/sobs/users/so_bb/cells_coadded_r0_Alens1_baseline_optimistic_231201.fits -O cells_coadded_cov_r0_Alens1_baseline_optimistic.fits

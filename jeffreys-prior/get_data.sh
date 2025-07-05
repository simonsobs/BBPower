#!/bin/bash -l

data_dir=/global/homes/k/kwolz/bbdev/BBPower/jeffreys-prior/data  ## YOUR DATA DIRECTORY

cd $data_dir

wget https://portal.nersc.gov/cfs/sobs/users/so_bb/cells_coadded_all_channels_gaussian_fgs.tar.gz
tar -xzvf cells_coadded_all_channels_gaussian_fgs.tar.gz -C $data_dir

wget https://portal.nersc.gov/cfs/sobs/users/so_bb/cells_coadded_r0_Alens1_baseline_optimistic_231201.fits -O cells_coadded_cov_r0_Alens1_baseline_optimistic.fits

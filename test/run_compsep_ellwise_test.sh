#!/bin/bash

mkdir -p test/test_out

# Generate fake data
python ./examples/generate_SO_spectra.py test/test_out

# Run fiducial compsep (Cori login node: 6 min)
python -m bbpower BBCompSep   --cells_coadded="./test/test_out/cls_coadd.fits"   --cells_noise="./test/test_out/cls_noise.fits"   --cells_fiducial="./test/test_out/cls_fid.fits"   --output_dir="./test/test_out"   --config_copy="./test/test_out/config_copy.yml"   --config="./test/test_config_emcee.yml"

# Run ellwise compsep (Cori login node: 119 min)
python -m bbpower BBCompSep   --cells_coadded="./test/test_out/cls_coadd.fits"   --cells_noise="./test/test_out/cls_noise.fits"   --cells_fiducial="./test/test_out/cls_fid.fits"   --output_dir="./test/test_out"   --config_copy="./test/test_out/config_copy_ellwise.yml"   --config="./test/test_config_ellwise.yml"

# Save and validate CMB bandpowers
python -m bbpower BBEllwise   --default_chain="./test/test_out/emcee.npz"   --output_dir="./test/test_out"   --config="./test/test_config_ellwise.yml"

# Cleanup
rm -r test/test_out
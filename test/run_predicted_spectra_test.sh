#!/bin/bash

# Create the output directory
mkdir -p test/test_out

# Generate some fake data
python ./examples/generate_SO_spectra.py test/test_out

# This computes the predicted spectra from the MAP parameters
python -m bbpower BBCompSep   --cells_coadded=./test/test_out/cls_coadd.fits   --cells_noise=./test/test_out/cls_coadd.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --cells_coadded_cov=./test/test_out/cls_coadd.fits   --output_dir=./test/test_out   --config_copy=./test/test_out/config_copy.yml   --config=./test/test_config_predicted_spectra.yml

# Check the predicted spectra exist
if [ ! -f ./test/test_out/cells_model.npz ]; then
    echo "Test did not pass"
else
    echo "Test passed."
fi

# Cleanup
rm -r test/test_out

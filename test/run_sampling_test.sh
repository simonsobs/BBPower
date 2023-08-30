#!/bin/bash

# Create the output directory
mkdir -p test/test_out

# Generate some fake data
python ./examples/generate_SO_spectra.py test/test_out

# Run pipeline
# This runs the likelihood sampler
python -m bbpower BBCompSep   --cells_coadded=./test/test_out/cls_coadd.fits   --cells_noise=./test/test_out/cls_coadd.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --cells_coadded_cov=./test/test_out/cls_coadd.fits   --param_chains=./test/test_out/param_chains.npz   --config_copy=./test/test_out/config_copy.yml   --config=./test/test_config_sampling.yml

# This plots the results of the pipeline
python -m bbpower BBPlotter   --cells_coadded_total=./test/test_out/cls_coadd.fits   --cells_coadded=./test/test_out/cls_coadd.fits   --cells_noise=./test/test_out/cls_coadd.fits   --cells_null=./test/test_out/cls_coadd.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --param_chains=./test/test_out/param_chains.npz   --plots=./test/test_out/plots.dir   --plots_page=./test/test_out/plots_page.html   --config=./test/test_config_sampling.yml

# Check the final plots exist
if [ ! -f ./test/test_out/plots.dir/triangle.png ]; then
    echo "Test did not pass"
else
    echo "Test passed"
fi

# Cleanup
rm -r test/test_out

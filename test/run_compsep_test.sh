#!/bin/bash

mkdir -p test/test_out

# Generate fake data
python ./examples/generate_SO_spectra.py test/test_out

# Run pipeline
python -m bbpower BBCompSep   --cells_coadded=./test/test_out/cls_coadd.fits   --cells_noise=./test/test_out/cls_noise.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --param_chains=./test/test_out/param_chains.npz   --config_copy=./test/test_out/config_copy.yml   --config=./test/test_config.yml

# Check chi2
if python -c "import numpy as np; chi2=np.load('test/test_out/param_chains.npz')['chi2']; assert chi2<1E-5"; then
    echo "Test passed"
else
    echo "Test did not pass"
fi

# Cleanup
rm -r test/test_out

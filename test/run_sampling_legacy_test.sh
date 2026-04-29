#!/bin/bash

# Legacy direct spectra -> BBCompSep -> BBPlotter smoke test.
# Current main-line tests use run_compsep_test.sh and run_power_specter_test.sh.

mkdir -p test/test_out

# Generate some fake data
python ./examples/generate_SO_spectra.py test/test_out

# Run component separation
python -m bbpower BBCompSep \
  --cells_coadded=./test/test_out/cls_coadd.fits \
  --cells_noise=./test/test_out/cls_noise.fits \
  --cells_fiducial=./test/test_out/cls_fid.fits \
  --cells_coadded_cov=./test/test_out/cls_coadd.fits \
  --output_dir=./test/test_out \
  --config_copy=./test/test_out/config_copy.yml \
  --config=./test/test_config_sampling_legacy.yml

# Plot the results
python -m bbpower BBPlotter \
  --cells_coadded_total=./test/test_out/cls_coadd.fits \
  --cells_coadded=./test/test_out/cls_coadd.fits \
  --cells_noise=./test/test_out/cls_noise.fits \
  --cells_null=./test/test_out/cls_coadd.fits \
  --cells_fiducial=./test/test_out/cls_fid.fits \
  --param_chains=./test/test_out/chi2.npz \
  --plots=./test/test_out/plots.dir \
  --plots_page=./test/test_out/plots_page.html \
  --config=./test/test_config_sampling_legacy.yml

if [ ! -f ./test/test_out/chi2.npz ] || [ ! -f ./test/test_out/plots_page.html ]; then
    echo "Test did not pass"
else
    echo "Test passed"
fi

rm -r test/test_out

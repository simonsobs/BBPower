#!/bin/bash

output_dir="test/david_theory_kevin_cov"
kevin_file="/global/homes/k/kwolz/bbdev/BBPower/examples/data/cells_coadded_r0_Alens1_baseline_optimistic_231201.fits"
david_file="/pscratch/sd/k/kwolz/bbdev/BBSims/output_simtest/sim_seed1001_cl_theory.fits"

# Create data outdir
mkdir -p "${output_dir}"

python -m bbpower BBCompSep \
    --cells_coadded="${david_file}" \
    --cells_noise="test/dummy.fits" \
    --cells_fiducial="test/dummy.fits" \
    --cells_coadded_cov="${kevin_file}" \
    --output_dir="${output_dir}" \
    --config_copy="${output_dir}/config_copy.yml" \
    --config="test/config_SO.yml"
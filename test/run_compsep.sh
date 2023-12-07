#!/bin/bash
################################################################################
### Init configs
cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/goal/optimistic/cells"
chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside512/full/r0_inhom-data/rfree-model/realistic/d0s0/goal/optimistic"
################################################################################

# Set data seed
data_seed='1001_david'

# Create data outdir
rm -rf "test/test_out_${data_seed}"
mkdir -p "test/test_out_${data_seed}"


# # Uses separate Cells and covariance to sample model posterior
# python -m bbpower BBCompSep \
#     --cells_coadded="${chainsdir}/test_out_${data_seed}/cells_coadded.fits" \
#     --cells_noise="${chainsdir}/test_out_${data_seed}/cells_noise.fits" \
#     --cells_fiducial="${cellsdir}/cls_fid.fits" \
#     --cells_coadded_cov="/global/homes/k/kwolz/bbdev/BBPower/examples/data/cells_coadded_r0_Alens1_baseline_optimistic_231201.fits" \
#     --output_dir="test/test_out_${data_seed}" \
#     --config_copy="test/test_out_${data_seed}/config_copy.yml" \
#     --config="test/config_SO.yml"
    

# Uses separate Cells and covariance to sample model posterior
python -m bbpower BBCompSep \
    --cells_coadded="/pscratch/sd/k/kwolz/bbdev/BBSims/output_simtest/sim_seed1001_cl_data.fits" \
    --cells_noise="/pscratch/sd/k/kwolz/bbdev/BBSims/output_simtest/sim_seed1001_cl_data.fits" \
    --cells_fiducial="/pscratch/sd/k/kwolz/bbdev/BBSims/output_simtest/sim_seed1001_cl_theory.fits" \
    --cells_coadded_cov="/global/homes/k/kwolz/bbdev/BBPower/examples/data/cells_coadded_r0_Alens1_baseline_optimistic_231201.fits" \
    --output_dir="test/test_out_${data_seed}" \
    --config_copy="test/test_out_${data_seed}/config_copy.yml" \
    --config="test/config_SO.yml"
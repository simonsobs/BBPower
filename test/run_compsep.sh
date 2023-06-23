#!/bin/bash
################################################################################
### Init configs
cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic/cells"
chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside512/full/r0_inhom-data/rfree-model/realistic/d0s0/baseline/optimistic"
################################################################################

# Set data seed
data_seed='0000'

# Create data outdir
mkdir -p "test/test_out_${data_seed}"

# Uses separate Cells and covariance to sample model posterior
python -m bbpower BBCompSep \
    --cells_coadded="${chainsdir}/test_out_${data_seed}/cells_coadded.fits" \
    --cells_noise="${chainsdir}/test_out_${data_seed}/cells_noise.fits" \
    --cells_fiducial="${cellsdir}/cls_fid.fits" \
    --cells_coadded_cov="${cellsdir}/cells_coadded.fits" \
    --output_dir="test_out_${data_seed}" \
    --config_copy="test_out_${data_seed}/config_copy.yml" \
    --config="test/config_SO.yml"
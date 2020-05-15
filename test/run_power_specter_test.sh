#!/bin/bash

mkdir -p test/test_out

# Generate fiducial cls
python ./examples/generate_SO_spectra.py test/test_out

# Generate simulations
for seed in {1001..1100}
do
    mkdir -p test/test_out/s${seed}
    echo ${seed}
    python examples/generate_SO_maps.py --output-dir test/test_out/s${seed} --seed ${seed} --nside 64
done

# Run pipeline
python -m bbpower BBPowerSpecter   --splits_list=./examples/test_data/splits_list.txt   --masks_apodized=./examples/test_data/masks_ones.fits.gz   --bandpasses_list=./examples/data/bpass_list.txt   --sims_list=./examples/test_data/sims_list.txt   --beams_list=./examples/data/beams_list.txt   --cells_all_splits=./test/test_out/cells_all_splits.fits   --cells_all_sims=./test/test_out/cells_all_sims.txt   --mcm=./test/test_out/mcm.dum   --config=./test/test_config.yml

python -m bbpower BBPowerSummarizer   --splits_list=./examples/test_data/splits_list.txt   --bandpasses_list=./examples/data/bpass_list.txt   --cells_fiducial=./test/test_out/cls_fid.fits   --cells_all_splits=./test/test_out/cells_all_splits.fits   --cells_all_sims=./test/test_out/cells_all_sims.txt   --cells_coadded_total=./test/test_out/cells_coadded_total.fits   --cells_coadded=./test/test_out/cells_coadded.fits   --cells_noise=./test/test_out/cells_noise.fits   --cells_null=./test/test_out/cells_null.fits   --config=./test/test_config.yml

python -m bbpower BBCompSep   --cells_coadded=./test/test_out/cells_coadded.fits   --cells_noise=./test/test_out/cells_noise.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --param_chains=./test/test_out/param_chains.npz   --config_copy=./test/test_out/config_copy.yml   --config=./test/test_config.yml

python -m bbpower BBPlotter   --cells_coadded_total=./test/test_out/cells_coadded_total.fits   --cells_coadded=./test/test_out/cells_coadded.fits   --cells_noise=./test/test_out/cells_noise.fits   --cells_null=./test/test_out/cells_null.fits   --cells_fiducial=./test/test_out/cls_fid.fits   --param_chains=./test/test_out/param_chains.npz   --plots=./test/test_out/plots.dir   --plots_page=./test/test_out/plots_page.html   --config=./test/test_config.yml

fchain="test/test_out/param_chains.npz"
if python -c "import numpy as np; a=np.load('${fchain}'); rchi2 = a['chi2'] / a['ndof']; print('chi2/dof = ', rchi2) ; assert rchi2 < 2"; then
    echo "Test passed"
else
    echo "Test did not pass"
fi

# Cleanup
rm -r test/test_out

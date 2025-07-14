# #!/bin/bash

sacc_file=/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_planck/sacc_files/cl_and_cov_sacc.fits
outdir=output_planck/bb_dust+synch+r_diag_cov_wide_prior
paramfile=paramfiles/paramfile_planck.yml
mkdir -p $outdir

# Generate fake data
python ./examples/generate_fiducial_spectra.py \
    --globals $paramfile \
    --outdir $outdir

python -m bbpower BBCompSep --cells_coadded=$sacc_file \
                            --cells_noise="${outdir}/cls_noise.fits" \
                            --cells_fiducial="${outdir}/cls_fid.fits" \
                            --cells_coadded_cov=$sacc_file \
                            --output_dir=$outdir \
                            --config_copy="${outdir}/config.yml" \
                            --config=$paramfile

python -m bbpower BBPlotter --cells_coadded_total="dummy.file" \
                            --cells_coadded=$sacc_file \
                            --cells_noise="dummy.file" \
                            --cells_null="dummy.file" \
                            --cells_fiducial="${outdir}/cls_fid.fits" \
                            --cells_best_fit="${outdir}/cells_best_fit.fits" \
                            --param_chains="${outdir}/emcee.npz" \
                            --chisq="${outdir}/chisq.npz" \
                            --config=$paramfile \
                            --plots="${outdir}/plots"

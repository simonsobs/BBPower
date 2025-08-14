# #!/bin/bash

# Nodes: salloc -N4 -C cpu -q interactive -t 04:00:00

bbpower_dir=/global/homes/k/kwolz/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER
basedir=/global/homes/k/kwolz/bbdev/BBPower/fisher_2025  ## YOUR RUNNING DIRECTORY
config="${basedir}/configs/spectra_wolzetal2024.yaml"

module load soconda

cd $basedir

## Uncomment steps as needed!

# Compute mode coupling matrix
# python -u ${bbpower_dir}/bbpower/power_specter_nopipe.py --config ${config} --do_mcm

# Compute cross-bundle spectra
#srun -n 100 -c 2 --cpu_bind=cores \
#python -u ${bbpower_dir}/bbpower/power_specter_nopipe.py --config ${config}

# Coadd cross-bundle spectra
#srun -n 100 -c 2 --cpu_bind=cores \
#python -u ${bbpower_dir}/bbpower/power_summarizer_nopipe.py --config ${config}

# Get covariance
python -u ${bbpower_dir}/bbpower/power_summarizer_nopipe.py --config ${config} --do_covar
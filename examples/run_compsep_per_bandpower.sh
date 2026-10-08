#!/bin/bash -l

## Software environment
export OMP_NUM_THREADS=3
export PYTHONPATH="${PYTHONPATH}:/home/kw6905/bbdev/BBPower"  # YOUR LOCAL BBPower INSTALL PATH

basedir=.  ## YOUR RUNNING DIRECTORY
bbpower_dir=.  ## PATH TO YOUR LOCAL BBPOWER
cd $basedir

bbpower_config=${basedir}/examples/config_nopipe_per_bandpower_model.yml
srun -n 32 -c 3 \
python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config
srun python -u ${bbpower_dir}/bbpower/plotter_per_bandpower_model.py --config $bbpower_config

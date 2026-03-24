#!/bin/bash -l

## Software environment
export OMP_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH}:/home/kw6905/bbdev/BBPower"  # YOUR LOCAL BBPower INSTALL PATH

basedir=/home/kw6905/bbdev/BBPower/examples  ## YOUR RUNNING DIRECTORY
bbpower_dir=/home/kw6905/bbdev/BBPower  ## PATH TO YOUR LOCAL BBPOWER
cd $basedir

bbpower_config=${basedir}/config_nopipe.yml
python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config
python -u ${bbpower_dir}/bbpower/plotter_nopipe.py --config $bbpower_config

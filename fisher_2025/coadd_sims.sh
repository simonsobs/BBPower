#!/bin/bash -l

# Nodes: salloc -N4 -C cpu -q interactive -t 04:00:00

basedir=/global/homes/k/kwolz/bbdev/BBPower/fisher_2025 ## YOUR RUNNING DIRECTORY

srun -n 250 -c 1 --cpu_bind=cores python -u ${basedir}/coadd_sims.py

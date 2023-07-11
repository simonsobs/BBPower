#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH --mail-user=kwolz@sissa.it
#SBATCH --mail-type=fail
#SBATCH -t 00:10:00
#SBATCH -o run_compsep_d1s1blop.log

# Tested 100 sims on 4 CPU nodes, preempt (wall clock time 6 min)
################################################################################
### Init configs
export config="test/config_SO.yml"
export cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic/cells"
export chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside512/full/r0_inhom-data/rfree-model/realistic/d1s1/baseline/optimistic"
################################################################################

# Load python environment
module load python
source activate bbpower
export OMP_NUM_THREADS=1

# Create the output directory
mkdir -p "${chainsdir}"

# Set data seeds
seeds=( $(seq -f "%04g" 0000 0499 ) )
printf "%s\n" "${seeds[@]}" > "${chainsdir}/seeds.txt" 
export nseeds=${#seeds[@]}

# Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks to run compsep stage"

# Calculate data-Cells and sample posteriors
> "${chainsdir}/out.log"
> "${chainsdir}/err.log"
echo "*** Logging to ${chainsdir}/out.log and ${chainsdir}/err.log ***"
srun "--ntasks=${ntasks}" "bash_parallel/bbpower_compsep.sh"
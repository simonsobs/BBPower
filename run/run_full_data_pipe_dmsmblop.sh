#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH --mail-user=kwolz@sissa.it
#SBATCH --mail-type=fail
#SBATCH -t 00:60:00
#SBATCH -o run_full_data_pipe_dmsmblop.log

# Load python environment
#module purge
module load python
#module load gsl
source activate bbpower

# Tested on 1 preempt node with 24 sims (22 mins)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_david_nside512.fits"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside512/full/apo-david"
export config="test/config_SO.yml"
export datadir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/realistic/dmsm/baseline/optimistic"
export cellsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic/cells"
export chainsdir="/pscratch/sd/k/kwolz/BBPower/chains/nside512/full/r0_inhom-data/rfree-model/realistic/dmsm/baseline/optimistic"
################################################################################

export OMP_NUM_THREADS=1

# Create the output directory
mkdir -p "${chainsdir}"

# Set data seeds
seeds=( $(seq -f "%04g" 0026 0075 ) )
printf "%s\n" "${seeds[@]}" > "${chainsdir}/seeds.txt" 
export nseeds=${#seeds[@]}

# Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks to run full data pipeline"

# Calculate data-C_ells and sample posteriors
> "${chainsdir}/out.log"
> "${chainsdir}/err.log"
echo "*** Logging to ${chainsdir}/out.log and ${chainsdir}/err.log ***"
srun "--ntasks=${ntasks}" "bash_parallel/bbpower_data.sh"
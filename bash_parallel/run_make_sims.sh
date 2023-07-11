#!/bin/bash
#SBATCH -N 15
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH --mail-user=kwolz@sissa.it
#SBATCH --mail-type=fail
#SBATCH -t 00:30:00
#SBATCH -o run_make_sims_blop.log

# Tested on 15 CPU nodes, preempt, with 500 sims (wall clock time 21 mins)
################################################################################
### Init configs
export splits="SO_SAT_obs_map_split_1of4.fits SO_SAT_obs_map_split_2of4.fits SO_SAT_obs_map_split_3of4.fits SO_SAT_obs_map_split_4of4.fits"
export mask_apodized="/pscratch/sd/k/kwolz/BBPower/examples/data/maps/mask_apodized_david_nside512.fits"
export config="test/config_SO.yml"
export mcmsdir="/pscratch/sd/k/kwolz/BBPower/mcms/nside512/full/apo-david"
export simsdir="/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic"
export cellsdir="${simsdir}/cells"
export sims_list="${cellsdir}/sims_list.txt"
################################################################################

# Load python environment
module load python
source activate bbpower
export OMP_NUM_THREADS=1

# Create the output directories
mkdir -p "${mcmsdir}"
mkdir -p "${cellsdir}"

# Define data splits (arbitrary seed 0000)
> "${cellsdir}/splits_list.txt"
for val in $splits; do
    echo "${simsdir}/0000/${val}" >> "${cellsdir}/splits_list.txt"
done

# Set simulation seeds
sim_seeds=( $(seq -f "%04g" 0000 0499 ) )
> $sims_list
for i in "${sim_seeds[@]}"; do
    echo "${simsdir}/${i}" >> $sims_list
done

# Set number of CPUs
ncpus=$(( $SLURM_CPUS_ON_NODE * $SLURM_JOB_NUM_NODES ))
ntasks=$(( $nseeds < $ncpus ? $nseeds : $ncpus ))
echo "Create ${ntasks} tasks to make simulations"

# Compute MCMs and simulation C_ells
> "${cellsdir}/out.log"
> "${cellsdir}/err.log"
echo "*** Logging to ${cellsdir}/out.log and ${cellsdir}/err.log ***"
srun "--ntasks=${ntasks}" "bash_parallel/bbpower_sims.sh"

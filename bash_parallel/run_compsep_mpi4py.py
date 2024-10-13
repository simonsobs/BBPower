import sys
sys.path.append("/global/homes/k/kwolz/bbdev/BBPower/bbpower")

import os
import mpi_utils as mpi
import compsep_nopipe
from argparse import Namespace

chainsdir = "/pscratch/sd/k/kwolz/BBPower/chains/nside256/full/r0_inhom-data/rfree-model/gaussian/baseline/optimistic"  # noqa
cellsdir = "/pscratch/sd/k/kwolz/BBPower/sims/nside256/full/r0_inhom/gaussian/baseline/optimistic/cells"  # noqa
config = "paramfiles/paramfile_SAT.yml"
Nsims = 100
Nstart = 0

mpi.init(True)

for id_sim in mpi.taskrange(Nsims):
    data_seed = str(id_sim + Nstart).zfill(4)
    print("data_seed", data_seed)

    bbpower_args = dict(
        cells_coadded=f"{chainsdir}/test_out_{data_seed}/cells_coadded.fits",
        cells_noise=f"{chainsdir}/test_out_{data_seed}/cells_noise.fits",
        cells_fiducial=f"{cellsdir}/cls_fid.fits",
        cells_coadded_cov=f"{cellsdir}/cells_coadded.fits",
        output_dir=f"{chainsdir}/test_out_{data_seed}",
        config_copy=f"{chainsdir}/test_out_{data_seed}/config_copy.yml",
        config=f"{config}"
    )

    compsep_nopipe.main(Namespace(**bbpower_args))

# Check if output files are there
for id_sim in range(Nsims):
    data_seed = str(id_sim + Nstart).zfill(4)
    if not os.path.isfile(
        f"{chainsdir}/test_out_{data_seed}/polychord/pch.txt"
    ):
        print(f"ERROR: seed {id_sim} is missing.")

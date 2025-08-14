import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from soopercool import mpi_utils as mpi
import os


# Overwrite sims that have already been written to disk.
overwrite = True

nside = 512
Alens = 1
plot_dir = "/global/homes/k/kwolz/bbdev/BBPower/fisher_2025/plots/wolzetal2024"
output_dir = "/pscratch/sd/k/kwolz/wolzetal2024"
mask_common = "/global/cfs/projectdirs/sobs/www/users/so_bb/apodized_mask_bbpipe_paper.fits"  # noqa
timeline = "NOMINAL_5YR"

noise_dir = "/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20210727/baseline_optimistic/{id_sim:04d}/SO_SAT_{freq}_noise_split_{id_bundle}of4_{id_sim:04d}_baseline_optimistic_20210727.fits"  # noqa
freqs = ["27", "39", "93", "145", "225", "280"]
ids_bundle = [1, 2, 3, 4]
ids_sim = [i for i in range(500)]

#sky_dir = "/global/cfs/cdirs/sobs/sims/so_extended/fg_sims/gaussian/{id_sim:04d}/sim_seed{seed}_{comp}_{band}.fits"  # noqa
sky_dir = {
    "sync": "/global/cfs/cdirs/sobs/users/krach/BBSims/FG_20201207/gaussian/foregrounds/synch/{id_sim:04d}/SO_SAT_{freq}_synch_{id_sim:04d}_gaussian_20201207.fits",
    "dust": "/global/cfs/cdirs/sobs/users/krach/BBSims/FG_20201207/gaussian/foregrounds/dust/{id_sim:04d}/SO_SAT_{freq}_dust_{id_sim:04d}_gaussian_20201207.fits",
    "cmb": "/global/cfs/cdirs/sobs/users/krach/BBSims/CMB_r0_20201207/cmb/{id_sim:04d}/SO_SAT_{freq}_cmb_{id_sim:04d}_CMB_r0_20201207.fits"
}

band_label = {
    "27": "band1", "39": "band2", "93": "band3",
    "145": "band4", "225": "band5", "280": "band6"
}

# sat_years = {
#     # SO extended (2025-2042):
#     # see https://docs.google.com/spreadsheets/d/1K3K6gFsSluHaA_vzLbVwwADZvonDWW2nmFKNsqfnfko/edit?gid=0#gid=0  # noqa
#     "so_extended": {
#         "030": 34.,
#         "040": 34.,
#         "090": 150.,
#         "150": 150.,
#         "230": 68.,
#         "290": 68.,
#     },
#     # Suzanne's timeline (2025-2034):
#     # see https://simonsobs.slack.com/archives/C096FGJ3VCG/p1753457851673079
#     "12SAT_10YR": {
#         "030": 2,  # 10 for 1 more LAT year from 2029 through 2034
#         "040": 2,  # 10 for 1 more LAT year from 2029 through 2034
#         "090": 59.33,
#         "150": 59.33,
#         "230": 21,
#         "290": 21,
#     },
#     "NOMINAL_4YR": {
#         "030": 1.5,
#         "040": 1.5,
#         "090": 11.33,
#         "150": 11.33,
#         "230": 3,
#         "290": 3,
#     }
#     "WOLZETAL_5YR": {
#         "030": 2,
#         "040": 2,
#         "090": 8,
#         "150": 8,
#         "230": 5,
#         "290": 5,
#     }
# }

# # To get the factor to multiply Reijo's sims with, do:
# # sqrt(sat_years_so_extended/sat_years_target)
# rescaling = {
#     freq: np.sqrt(sat_years["so_extended"][freq]/sat_years["NOMINAL_4YR"][freq])
#     for freq in freqs
# }

# # Print analysis masks
common_masks = {}
binary_masks = {}
surveys = ["wolzetal2024"]
for survey in surveys:
    common_masks[survey] = hp.read_map(mask_common)
    binary_masks[survey] = np.array(common_masks[survey] > 0, np.float32)
    if not os.path.isfile(f"{output_dir}/bbpower_input/mask_common_{survey}.fits"):  # noqa
        hp.write_map(
            f"{output_dir}/bbpower_input/mask_common_{survey}.fits",
            common_masks[survey], overwrite=False, dtype=np.float32
        )
    print(f"{output_dir}/bbpower_input/mask_common_{survey}.fits")


# MPI related initialization
rank, size, comm = mpi.init(True)


mpi_shared_list = [(id_sim, survey)
                   for id_sim in ids_sim
                   for survey in surveys]

# Every rank must have the same shared list
mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                logger=None)
local_mpi_list = [mpi_shared_list[i] for i in task_ids]

# # Loop over sims and survey types
for id_sim, survey in local_mpi_list:
    print("sim #", id_sim, survey)
    if not overwrite and os.path.isdir(f"{output_dir}/coadded_sims/gaussian/Alens{Alens}/{survey}/{id_sim:04d}"):  # noqa
        continue

    # Read & rescale noise sims
    print("Reading noise sims")
    noise_sims = {
        (freq, id_bundle): hp.read_map(
            noise_dir.format(id_sim=id_sim,
                             freq=freq,
                             id_bundle=id_bundle),
            field=(1, 2)  # Ignore T noise
        )
        for id_bundle in ids_bundle
        for freq in freqs
    }
    if id_sim == 0:
        hp.mollview(noise_sims["93", 1][0, :], min=-10, max=10)
        plt.savefig(f"{plot_dir}/noiseQ_{id_sim:04d}_{survey}_93.png")
        print(f"{plot_dir}/noiseQ_{id_sim:04d}_{survey}_93.png")
        plt.close()
        plt.clf()

    # # Load sky sims
    print("Reading sky sims")
    try:
        sky_sims = {
            (comp, freq): hp.read_map(
                sky_dir[comp].format(
                    id_sim=id_sim, freq=freq
                ),
                field=(1, 2)  # Ignore T
            )
            for freq in freqs
            for comp in ["cmb", "sync", "dust"]
        }
    except FileNotFoundError:
        print("Sky sim not found.")
        continue

    if id_sim == 0:
        for freq in ["27", "145"]:
            for comp in ["cmb", "sync", "dust"]:
                hp.mollview(sky_sims[comp, freq][0, :], min=-10, max=10)
                plt.savefig(f"{plot_dir}/{comp}Q_{id_sim:04d}_{freq}.png")
                print(f"{plot_dir}/{comp}Q_{id_sim:04d}_{freq}.png")
                plt.clf()

    # # Coadd signal and noise splits (outside binary mask)
    print("Coadding signal and noise sims")
    for id_bundle in ids_bundle:
        coadd = np.zeros((len(freqs), 3, hp.nside2npix(nside)))
        for id_freq, freq in enumerate(freqs):
            coadd[id_freq, 1:] = np.sqrt(Alens) * sky_sims["cmb", freq]
            for comp in ["dust", "sync"]:
                coadd[id_freq, 1:] += sky_sims[comp, freq]
            coadd[id_freq, 1:] += noise_sims[freq, id_bundle]
        # null sky pixels outside patch
        coadd[:, 1:, :] *= binary_masks[survey][None, None, :]

        dirname = f"{output_dir}/coadded_sims/gaussian/Alens{Alens}/{timeline}/{survey}/{id_sim:04d}"  # noqa
        fname = f"SO_SAT_obs_map_split_{id_bundle}of4.fits"

        if id_sim == 0 and id_bundle == 1:
            hp.mollview(coadd[2, 1, :], min=-10, max=10)
            plt.savefig(f"{plot_dir}/coaddQ_0000_{survey}_93.png")
            print(f"{plot_dir}/coaddQ_0000_{survey}_93.png")
            plt.clf()

        os.makedirs(dirname, exist_ok=True)
        hp.write_map(
            f"{dirname}/{fname}",
            coadd.reshape(len(freqs)*3, hp.nside2npix(nside)),
            overwrite=True,
            dtype=np.float32
        )
        print(f"WRITTEN  {dirname}/{fname}")

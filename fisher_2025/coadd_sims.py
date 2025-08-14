import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from soopercool import mpi_utils as mpi
import os


# Overwrite sims that have already been written to disk.
overwrite = True

nside = 512
Alens = 0.3
plot_dir = "/global/homes/k/kwolz/bbdev/BBPower/fisher_2025/plots"
output_dir = "/global/cfs/cdirs/sobs/sims/fisher_2025"
mask_common = "/global/cfs/cdirs/sobs/sims/so_extended/masks/masks_megatop/masks_baseline_{survey}/common_analysis_mask.fits"  # noqa
timeline = "NOMINAL_4YR"

noise_dir = "/global/cfs/cdirs/sobs/sims/so_extended/noise_sims/sat_noise_splits/{sat_noise}/pessimistic/s{id_sim:04d}/noise_SAT_f{noise_freq}.{sat_noise}.pessimistic.{survey}{id_sim:04d}.splt{id_bundle}.fits"  # noqa
noise_levels = {"030": "goal", "040": "goal", "090": "goal", "150": "goal", "230": "baseline", "290": "baseline"}  # noqa
noise_freqs = {"030": "030", "040": "040", "090": "090", "150": "150", "230": "220", "290": "280"}  # noqa
surveys = {"fsky6": "deep.", "fsky14": ""}
freqs = ["030", "040", "090", "150", "230", "290"]
ids_bundle = [1, 2, 3, 4]
ids_sim = [i for i in range(500)]

sky_dir = "/global/cfs/cdirs/sobs/sims/so_extended/fg_sims/gaussian/{id_sim:04d}/sim_seed{seed}_{comp}_{band}.fits"  # noqa

band_label = {
    "030": "band1", "040": "band2", "090": "band3",
    "150": "band4", "230": "band5", "290": "band6"
}

sat_years = {
    # SO extended (2025-2042):
    # see https://docs.google.com/spreadsheets/d/1K3K6gFsSluHaA_vzLbVwwADZvonDWW2nmFKNsqfnfko/edit?gid=0#gid=0  # noqa
    "so_extended": {
        "030": 34.,
        "040": 34.,
        "090": 150.,
        "150": 150.,
        "230": 68.,
        "290": 68.,
    },
    # Suzanne's timeline (2025-2034):
    # see https://simonsobs.slack.com/archives/C096FGJ3VCG/p1753457851673079
    "12SAT_10YR": {
        "030": 2,  # 10 for 1 more LAT year from 2029 through 2034
        "040": 2,  # 10 for 1 more LAT year from 2029 through 2034
        "090": 59.33,
        "150": 59.33,
        "230": 21,
        "290": 21,
    },
    "NOMINAL_4YR": {
        "030": 1.5,
        "040": 1.5,
        "090": 11.33,
        "150": 11.33,
        "230": 3,
        "290": 3,
    }
}

# To get the factor to multiply Reijo's sims with, do:
# sqrt(sat_years_so_extended/sat_years_target)
rescaling = {
    freq: np.sqrt(sat_years["so_extended"][freq]/sat_years["NOMINAL_4YR"][freq])
    for freq in freqs
}

# # Print analysis masks
common_masks = {}
binary_masks = {}
patch = {"fsky6": "deep", "fsky14": "wide"}
for survey in surveys:
    common_masks[survey] = hp.read_map(
        mask_common.format(survey=patch[survey])
    )
    binary_masks[survey] = np.array(common_masks[survey] > 0, np.float64)
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
            noise_dir.format(id_sim=id_sim, survey=surveys[survey],
                             noise_freq=noise_freqs[freq],
                             sat_noise=noise_levels[freq],
                             id_bundle=id_bundle),
            field=(1, 2)  # Ignore T noise
        ) * rescaling[freq]
        for id_bundle in ids_bundle
        for freq in freqs
    }
    if id_sim == 0:
        hp.mollview(noise_sims["090", 1][0, :], min=-10, max=10)
        plt.savefig(f"{plot_dir}/noiseQ_{id_sim:04d}_{survey}_f090.png")
        print(f"{plot_dir}/noiseQ_{id_sim:04d}_{survey}_f090.png")
        plt.close()
        plt.clf()

    # # Load sky sims
    print("Reading sky sims")
    try:
        sky_sims = {
            (comp, freq): hp.read_map(
                sky_dir.format(
                    id_sim=id_sim, comp=comp, seed=id_sim+1000,
                    band=band_label[freq]
                ),
                field=(0, 1)  # sky sims only contain Q and U
            )
            for freq in freqs
            for comp in ["cmb", "sync", "dust", "sky"]
        }
    except FileNotFoundError:
        print("Sky sim not found.")
        continue

    if id_sim == 0:
        for freq in ["090", "150"]:
            for comp in ["cmb", "sync", "dust", "sky"]:
                hp.mollview(sky_sims[comp, freq][0, :], min=-10, max=10)
                plt.savefig(f"{plot_dir}/{comp}Q_{id_sim:04d}_f{freq}.png")
                print(f"{plot_dir}/{comp}Q_{id_sim:04d}_f{freq}.png")
                plt.clf()

            # Double check that sky = cmb + dust + sync
            diff = sky_sims["sky",freq][0,:] - sky_sims["sync",freq][0,:] - sky_sims["dust",freq][0,:]  - sky_sims["cmb",freq][0,:]  # noqa
            hp.mollview(diff, min=-1E-6, max=1E-6)
            plt.savefig(f"{plot_dir}/diffQ_0000_f{freq}.png")
            print(f"{plot_dir}/diffQ_0000_f{freq}.png")
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
            plt.savefig(f"{plot_dir}/coaddQ_0000_{survey}_f090.png")
            print(f"{plot_dir}/coaddQ_0000_{survey}_f090.png")
            plt.clf()

        os.makedirs(dirname, exist_ok=True)
        hp.write_map(
            f"{dirname}/{fname}",
            coadd.reshape(len(freqs)*3, hp.nside2npix(nside)),
            overwrite=True,
            dtype=np.float32
        )
        print(f"WRITTEN  {dirname}/{fname}")

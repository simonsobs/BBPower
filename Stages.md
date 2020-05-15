Pipeline stages
---------------

BBPower is made up of the following stages:

## 1. BBPowerSpecter
### Inputs
- `splits_list`: a txt file containing a list of file paths. Each file should be a fits file containing 2 maps (Q and U) per frequency channel. Each file should correspond to one data split (i.e. a map made from only part of the data).
- `masks_apodized`: a fits file containing one map per frequency channel. Each map should be the apodized mask to be used on all maps corresponding to that frequency channel.
- `bandpasses_list`: a txt file containing a list of file paths. Each file should be a txt file containing the bandpass for one of the frequency channels (in the same order in which the maps are stored in the files above). Two columns: frequency and transmission.
- `sims_list`: a txt file containing a list of paths. Each path should lead to a directory containing a sky simulation. The simulation maps should be structured in the same way as the data splits contained in `splits_list`.
- `beams_list`: a txt file containing a list of file paths. Each file should be a txt file containing the beam for one of the frequency channels (in the same order in which the maps are stored in the files above). Two columns: multipole ell and beam_ell.

### Outputs
- `cells_all_splits`: a fits file containing all possible auto- and cross-spectra between all different data splits, frequency channels and polarization components (E and B). The files are in [SACC](https://github.com/LSSTDESC/sacc) format, and contain also the bandpass and beam information, as well as all the bandpower window functions.
- `cells_all_sims`: a txt file containing a list of files. Each file is the fits file containing the power spectra for a given simulation, in the same format as `cells_all_splits`.

### Parameters
- `bpw_edges`: a txt file containing a list of integers defining the edges of the bandpowers.
- `purify_B`: set to True if you want to use B-mode purification.
- `n_iter`: iter parameter for spherical harmonic transforms.


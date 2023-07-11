import numpy as np
import os, sys
import healpy as hp
from combine_noise import compute_noise_factors, combine_noise_maps

"""
Script to combine simulations into observed sky splits. Will save a signal only 
file called SO_SAT_maps_sky_signal.fits and also save 4 obs splits called 
SO_SAT_obs_map_split_kof4.fits. All maps are saved as healpix fits files with 18
maps, corresponding to TQU for 6 frequencies. 
Use .reshape(6, 3, -1) to get into frequency, TQU, npix shape. 
DO NOT USE T.
To parse:
* simsdir, e.g. '/global/cscratch1/sd/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic'
* seed:    4-digit zero-padded number between 0000 and 0499. Choose deterministically from cmb, fg and noise sims 
"""

def get_sky_signals(nrs, r001, Alens05, foreground, nside, freq_labels, fdir, fgnames):
    skymaps = np.zeros((6, 3, hp.nside2npix(nside)))
    for k, fl in enumerate(freq_labels):
        if r001 and not Alens05: # r=0.001
            fname = f'{fdir}/CMB_r_Alens_20211108/r001_Alens1/cmb/{nrs}/SO_SAT_{fl}_cmb_{nrs}_CMB_r001_Alens1_20211108.fits'   
        elif Alens05 and not r001: # Alens=0.5
            fname = f'{fdir}/CMB_r_Alens_20211108/r0_Alens05/cmb/{nrs}/SO_SAT_{fl}_cmb_{nrs}_CMB_r0_Alens05_20211108.fits' 
        elif Alens05 and r001: # r=0.001 and Alens=0.5
            fname = f'{fdir}/CMB_r_Alens_20211108/r001_Alens05/cmb/{nrs}/SO_SAT_{fl}_cmb_{nrs}_CMB_r001_Alens05_20220516.fits' 
        else: # r=0 and Alens=1 (default)
            fname = f'{fdir}/CMB_r0_20201207/cmb/{nrs}/SO_SAT_{fl}_cmb_{nrs}_CMB_r0_20201207.fits'  
        skymaps[k] += hp.read_map(fname, field=range(3), verbose=False)
        for fgn in fgnames:
            if foreground == 'gaussian':
                fname = f'{fdir}/FG_20201207/gaussian/foregrounds/{fgn}/{nrs}/SO_SAT_{fl}_{fgn}_{nrs}_gaussian_20201207.fits'
            elif foreground == 'd10s5': # d10s5
                fname = f'{fdir}/FG_20220516/d10s5/foregrounds/{fgn}/SO_SAT_{fl}_{fgn}_d10s5_20220516.fits'
            else: # d0s0, d1s1, dmsm
                fname = f'{fdir}/FG_20201207/realistic/{foreground}/foregrounds/{fgn}/SO_SAT_{fl}_{fgn}_{foreground}_20201207.fits'
            skymaps[k] += hp.read_map(fname, field=range(3), verbose=False)
    return skymaps

def get_noise(nrs, sp, sensitivity_mode, one_over_f, inhom, fdir_inhom, nside, freq_labels, mask=None):
    noisemaps = np.zeros((6, 3, hp.nside2npix(nside)))
    if not inhom:
        factors = compute_noise_factors(sensitivity_mode, one_over_f)
    for k, fl in enumerate(freq_labels):
        if inhom:
            if sensitivity_mode == 2:
                if one_over_f == 1:
                    if mask=='full-opt-al1': # new lensing-optimized mask
                        fname = f'/global/cfs/cdirs/sobs/users/emilie_h/Noise_sims_May2023/goal_optimistic/{nrs}/SO_SAT_{fl}_noise_split_{sp+1}of4_{nrs}_goal_optimistic.fits'
                    else: # new goal white noise level
                        fname = f'/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20230531/goal_optimistic/{nrs}/SO_SAT_{fl}_noise_split_{sp+1}of4_{nrs}_goal_optimistic_20230531.fits'
                elif one_over_f == 0:
                    fname = f'{fdir_inhom}/goal_pessimistic/{nrs}/SO_SAT_{fl}_noise_split_{sp+1}of4_{nrs}_goal_pessimistic_20210727.fits'
            elif sensitivity_mode == 1:
                if one_over_f == 1:
                    fname = f'{fdir_inhom}/baseline_optimistic/{nrs}/SO_SAT_{fl}_noise_split_{sp+1}of4_{nrs}_baseline_optimistic_20210727.fits'
                elif one_over_f == 0:
                    fname = f'{fdir_inhom}/baseline_pessimistic/{nrs}/SO_SAT_{fl}_noise_split_{sp+1}of4_{nrs}_baseline_pessimistic_20210727.fits'
            noisemaps[k] += hp.read_map(fname, field=range(3), verbose=False)
        else:
            noisemaps[k] = combine_noise_maps(int(nrs) + sp*500, int(fl), factors)
            noisemaps[k, 1, :] *= np.sqrt(4.)
            noisemaps[k, 2, :] *= np.sqrt(4.)
    return noisemaps

def get_path(sdir, nrs, sensitivity_mode, one_over_f, foreground):
    if foreground == 'gaussian':
        path = os.path.join(sdir, 'gaussian')
    else:
        path = os.path.join(sdir, 'realistic')
        path = os.path.join(path, foreground)

    if sensitivity_mode == 1: 
        path = os.path.join(path, 'baseline')
    else: 
        path = os.path.join(path, 'goal')

    if one_over_f is None: 
        path = os.path.join(path, 'white')
    elif one_over_f == 0: 
        path = os.path.join(path, 'pessimistic')
    else:   
        path = os.path.join(path, 'optimistic')

    path = os.path.join(path, nrs, '')

    try:
        os.mkdir(path)
    except:
        pass
    print(path)
    return path

def run(sdir, nrs, r001, Alens05, sensitivity_mode, one_over_f, foreground, inhom, freq_labels,    # NEW
        fdir, fdir_inhom, fgnames, overwrite=False, nside=512):
    """ Create signal maps and obs sky maps for all 6 SO frequencies. Write to 
        disk. example params: 
        sdir = 'test/'              # directory to save sims
        nrs = 0000                  # zero-padded realization number 0000-0499. 
        sensitivity_mode = 2        # noise sensitivity mode: 1 baseline, 2 goal
        one_over_f = None           # one over f: None no 1/f, 
                                    # 0 pessimistic 1/f, 1 optimistic 1/f
        foreground = 'gaussian'     # FG type. Choose from 'gaussian', 'd0s0', 
                                    # 'd1s1', or 'dmsm'.
        inhom = False               # boolean to indicate if inhomogeneous noise
                                    # is used
        freq_labels = ['27', '39', '93', '145', '225', '280']
        fdir = '/global/cfs/cdirs/sobs/users/krach/BBSims' # directory of noise maps
        fdir_inhom = '/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20210727'
        overwrite = False           # overwrite existing files
    """
    sdir = get_path(sdir, nrs, sensitivity_mode, one_over_f, foreground)
    from pathlib import Path
    Path(sdir).mkdir(parents=True, exist_ok=True)

    skymaps = get_sky_signals(nrs, r001, Alens05, foreground, nside, 
                              freq_labels, fdir, fgnames)
    sname = f'{sdir}SO_SAT_maps_sky_signal.fits'
    mask = None
    if 'full-opt-al1' in sdir:
        mask = 'full-opt-al1'
    if overwrite==False and os.path.isfile(sname)==False:
        hp.write_map(sname, skymaps.reshape(18, -1))
    for sp in range(4):
        noisemaps = get_noise(nrs, sp, sensitivity_mode, one_over_f, inhom, 
                              fdir_inhom, nside, freq_labels, mask=mask)
        sname = f'{sdir}SO_SAT_obs_map_split_{sp+1}of4.fits'
        totalmaps = noisemaps + skymaps
        if overwrite==False and os.path.isfile(sname)==False:
            hp.write_map(sname, totalmaps.reshape(18, -1))
    return 


def main(simsdir, seed):
    # To parse:
    # simsdir, e.g. '/pscratch/sd/k/kwolz/BBPower/sims/nside512/full/r0_inhom/gaussian/baseline/optimistic'
    # seed:    4-digit zero-padded number between 0 and 499
    oof = simsdir.split('/')[-1]
    sens = simsdir.split('/')[-2]
    fg = simsdir.split('/')[-3]
    inhom = False
    band = False
    sky = 'full'
    if ('north' in simsdir.split('/')[-5]) or ('north' in simsdir.split('/')[-6]):
        sky = 'north'
    if ('south' in simsdir.split('/')[-5]) or ('south' in simsdir.split('/')[-6]):
        sky = 'south' 
    if ('full-opt-al1' in simsdir.split('/')[-5]) or ('full-opt-al1' in simsdir.split('/')[-6]):
        sky = 'full-opt-al1' # Optimized mask ~1/(noise+temp_lensing) (AL=1 case)
    if ('inhom' in simsdir.split('/')[-4]) or ('inhom' in simsdir.split('/')[-5]):
        inhom = True
    if ('band' in simsdir.split('/')[-4]) or ('band' in simsdir.split('/')[-5]):
        band = True
    r001 = False
    if ('r001' in simsdir.split('/')[-4]) or ('r001' in simsdir.split('/')[-5]):
        r001 = True
    Alens05 = False                                                              
    if ('Alens05' in simsdir.split('/')[-4]) or ('Alens05' in simsdir.split('/')[-5]):
        Alens05 = True

    nside = 512
    freq_labels = ['27', '39', '93', '145', '225', '280']
    fgnames = ['synch', 'dust']
    fdir = '/global/cfs/cdirs/sobs/users/krach/BBSims'
    fdir_inhom = '/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20210727'
    sdir = '/pscratch/sd/k/kwolz/BBPower/sims/nside512/' # need trailing "/"
    sdir += sky + '/r0'
    if r001:                              
        sdir += '01'
    if Alens05:            
        sdir += '_Alens05'
    if inhom:
        sdir += '_inhom'
    if band:
        sdir += '_band'
    
    overwrite = False

    # Write maps to files
    sens_idx = {}
    sens_idx['baseline'] = 1
    sens_idx['goal'] = 2
    oof_idx = {}
    oof_idx['pessimistic'] = 0
    oof_idx['optimistic'] = 1

    run(sdir, seed, r001, Alens05, sens_idx[sens], oof_idx[oof], fg, inhom,
        freq_labels, fdir, fdir_inhom, fgnames, overwrite, nside)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

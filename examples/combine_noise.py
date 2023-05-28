import healpy as hp
import numpy as np
import V3_calc_public as sonc

def compute_noise_factors(sensitivity_mode, one_over_f):
    """
    returns the factors needed to combine white and one_over_f noise maps
    **Parameters**
    - `sensitivity_mode`: 
         0: threshold,
         1: baseline,
         2: goal
    - `one_over_f`:
         None: no one_over_f
         0: pessimistic
         1: optimistic
    **Returns**
    - the coefficients which needs to be multiplied to the noise maps in order to combine them
    """

    freqs = sonc.Simons_Observatory_V3_SA_bands()
    S_SA_27  = np.array([32,21,15])    
    S_SA_39  = np.array([17,13,10])    
    S_SA_93  = np.array([4.6,3.4,2.4]) 
    S_SA_145 = np.array([5.5,4.3,2.7])
    S_SA_225 = np.array([11,8.6,5.7])  
    S_SA_280 = np.array([26,22,14])
    wn_level = np.array([S_SA_27, S_SA_39, S_SA_93, S_SA_145, S_SA_225, S_SA_280])
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improving possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    f_knee_pol = np.array([f_knee_pol_SA_27, f_knee_pol_SA_39, f_knee_pol_SA_93, 
                           f_knee_pol_SA_145, f_knee_pol_SA_225, f_knee_pol_SA_280])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])
    factors_tot = []
    for nch in range(len(freqs)):
        wn_levels_orig = wn_level[nch, 1]
        wn_level_target = wn_level[nch, sensitivity_mode]
        wn_factor = (wn_level_target/wn_levels_orig)**2
        if one_over_f==1 or one_over_f==0:
            f_knee_target = f_knee_pol[nch, one_over_f]
            oof_factor = (wn_level_target/wn_levels_orig)**2.*(1/f_knee_target)**alpha_pol[nch]
        else:
            oof_factor = 0.
        factor = [wn_factor, oof_factor]
        factors_tot.append(np.sqrt(factor))
    return factors_tot

def combine_noise_maps(nmc, freq, noise_factors):
    """
    combines white and one_over_f noise maps
    **Parameters**
    - `nmc`: 
        montecarlo realization to be considered (int)
    - `freqs`:
        SAT frequency to be considered (int)
    - `noise_factors`:
        coefficients computed with the `compute_noise_factors`functions
    **Returns**
    - combined noise maps (TQU map)
    """
    freqs = sonc.Simons_Observatory_V3_SA_bands()
    freq_indx = int(np.where(freqs==freq)[0])
    freq_str = str(int(freq))
    nmc_str = str(nmc).zfill(4)
    dir_white = '/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/white/noise/'
    dir_oof = '/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/one_over_f/noise/'
    white_map = hp.read_map(f'{dir_white}{nmc_str}/SO_SAT_{freq_str}_noise_FULL_{nmc_str}_white_20201207.fits', 
                            (0,1,2), verbose=False)
    oof_map = hp.read_map(f'{dir_oof}{nmc_str}/SO_SAT_{freq_str}_noise_FULL_{nmc_str}_oof_20201207.fits', 
                          (0,1,2), verbose=False)
    factor = noise_factors[freq_indx]
    tot_map = white_map*factor[0]+oof_map*factor[1]
    return tot_map
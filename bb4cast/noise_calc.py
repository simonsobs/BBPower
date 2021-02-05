from __future__ import print_function
import numpy as np


def so_SAT_beams(ls):
    fwhm = np.array([91., 63., 30., 17., 11., 9.])
    sig = fwhm*np.pi/(np.sqrt(8.*np.log(2))*180.*60.)
    return np.array([np.exp(-0.5*ls*(ls+1)*sig**2)
                     for s in sig])


def so_SAT_Nl(info, ell_max, include_kludge=True):
    """
    sensitivity_mode
         1: baseline,
         2: goal
    one_over_f_mode
         0: pessimistic
         1: optimistic
    """
    sens_dict = {'baseline': 1, 'goal': 2}
    oof_dict = {'pessimistic': 0, 'optimistic': 1}
    sensitivity_mode = sens_dict[info.get('sensitivity', 'baseline')]
    one_over_f_mode = oof_dict[info.get('one_over_f', 'optimistic')]
    ydet_LF = info.get('ydet_LF', 1.)
    ydet_MF = info.get('ydet_MF', 9.)
    ydet_UHF = info.get('ydet_UHF', 5.)
    f_sky = info.get('f_sky', 0.1)
    S_SA_27 = np.array([1.e9, 21, 15]) * np.sqrt(1./ydet_LF)
    S_SA_39 = np.array([1.e9, 13, 10]) * np.sqrt(1./ydet_LF)
    # Factor 2 because numbers assumed a default 2 MF tubes
    S_SA_93 = np.array([1.e9, 3.4, 2.4]) * np.sqrt(2./ydet_MF)
    S_SA_145 = np.array([1.e9, 4.3, 2.7]) * np.sqrt(2./ydet_MF)
    S_SA_225 = np.array([1.e9, 8.6, 5.7]) * np.sqrt(1./ydet_UHF)
    S_SA_280 = np.array([1.e9, 22, 14]) * np.sqrt(1./ydet_UHF)
    f_knee_pol_SA_27 = np.array([30., 15.])
    f_knee_pol_SA_39 = np.array([30., 15.])  # from QUIET
    f_knee_pol_SA_93 = np.array([50., 25.])
    f_knee_pol_SA_145 = np.array([50., 25.])  # from ABS
    f_knee_pol_SA_225 = np.array([70., 35.])
    f_knee_pol_SA_280 = np.array([100., 40.])
    alpha_pol = np.array([-2.4, -2.4, -2.5, -3, -3, -3])

    t = 365. * 24. * 3600  # 1Y in s
    t = t * 0.2  # retention after observing efficiency and cuts
    if include_kludge:
        t = t * 0.85  # a kludge to account for map edges
    A_SR = 4 * np.pi * f_sky  # sky area in steradians

    ell = np.arange(ell_max+1)
    # Avoid division by zero
    ell[0] = 1

    # White
    W_T_27 = S_SA_27[sensitivity_mode] / np.sqrt(t)
    W_T_39 = S_SA_39[sensitivity_mode] / np.sqrt(t)
    W_T_93 = S_SA_93[sensitivity_mode] / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)

    # 1/f
    AN_P_27 = (ell / f_knee_pol_SA_27[one_over_f_mode])**alpha_pol[0] + 1.
    AN_P_39 = (ell / f_knee_pol_SA_39[one_over_f_mode])**alpha_pol[1] + 1.
    AN_P_93 = (ell / f_knee_pol_SA_93[one_over_f_mode])**alpha_pol[2] + 1.
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.

    # combined
    N_ell_P_27 = (W_T_27 * np.sqrt(2))**2. * A_SR * AN_P_27
    N_ell_P_39 = (W_T_39 * np.sqrt(2))**2. * A_SR * AN_P_39
    N_ell_P_93 = (W_T_93 * np.sqrt(2))**2. * A_SR * AN_P_93
    N_ell_P_145 = (W_T_145 * np.sqrt(2))**2. * A_SR * AN_P_145
    N_ell_P_225 = (W_T_225 * np.sqrt(2))**2. * A_SR * AN_P_225
    N_ell_P_280 = (W_T_280 * np.sqrt(2))**2. * A_SR * AN_P_280
    N_ell_P_SA = np.array([N_ell_P_27, N_ell_P_39, N_ell_P_93,
                           N_ell_P_145, N_ell_P_225, N_ell_P_280])
    N_ell_P_SA[:, :2] = 0
    ell[0] = 0

    return ell, N_ell_P_SA

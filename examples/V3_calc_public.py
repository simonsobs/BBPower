from __future__ import print_function
import numpy as np

####################################################################
####################################################################
### SAC CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_SA_beams():
    ## returns the SAC beams in arcminutes
    beam_SAC_27 = 91.
    beam_SAC_39 = 63.
    beam_SAC_93 = 30.
    beam_SAC_145 = 17.
    beam_SAC_225 = 11.
    beam_SAC_280 = 9.
    return(np.array([beam_SAC_27,beam_SAC_39,beam_SAC_93,beam_SAC_145,beam_SAC_225,beam_SAC_280]))

def Simons_Observatory_V3_SA_noise(sensitivity_mode,one_over_f_mode,SAC_yrs_LF,f_sky,ell_max,delta_ell):
    ## retuns noise curves, including the impact of the beam for the SO small aperture telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     -1: no white noise
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     -1: no one_over_f
    #     0: pessimistic
    #     1: optimistic
    # SAC_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAC
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERTURE
    # ensure valid parameter choices
    assert( sensitivity_mode == 0 or sensitivity_mode == 1 or sensitivity_mode == 2 or sensitivity_mode == -1)
    assert( one_over_f_mode == 0 or one_over_f_mode == 1 or one_over_f_mode == -1)
    assert( SAC_yrs_LF <= 5) #N.B. SAC_yrs_LF can be negative
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # configuration
    if sensitivity_mode == -1:
        sensitivity_mode = 1
        no_white_component = True
    else:
        no_white_component = False

    if (SAC_yrs_LF > 0):
        NTubes_LF  = SAC_yrs_LF/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 - SAC_yrs_LF/5.
    else:
        NTubes_LF  = np.fabs(SAC_yrs_LF)/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2
    NTubes_UHF = 1.
    # sensitivity
    # N.B. divide-by-zero will occur if NTubes = 0
    # handle with assert() since it's highly unlikely we want any configurations without >= 1 of each tube type
    assert( NTubes_LF > 0. )
    assert( NTubes_MF > 0. )
    assert( NTubes_UHF > 0.)
    S_SA_27  = np.array([32,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([17,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([4.6,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([5.5,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([11,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([26,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improving possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])  ## roughly consistent with Yuji's table, but extrapolated

    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    #t = t* 0.85  ## a kluge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky areas in Steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    print("sky area: ", A_deg, "degrees^2")
    print("when generating realizations from a hits map, the total integration time should be 1/0.85 longer")
    print("since we should remove a kluge for map non-uniformity since this is included correcly in a hits map")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(0, ell_max, delta_ell)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    print('sensitivity_mode', sensitivity_mode)
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)

    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    if no_white_component:
        AN_P_27  = (ell)**alpha_pol[0]
        AN_P_39  = (ell)**alpha_pol[1]
        AN_P_93  = (ell)**alpha_pol[2]
        AN_P_145 = (ell)**alpha_pol[3]
        AN_P_225 = (ell)**alpha_pol[4]
        AN_P_280 = (ell)**alpha_pol[5]

    elif one_over_f_mode==-1:
        AN_P_27  = np.ones(len(ell))
        AN_P_39  = np.ones(len(ell))
        AN_P_93  = np.ones(len(ell))
        AN_P_145 = np.ones(len(ell))
        AN_P_225 = np.ones(len(ell))
        AN_P_280 = np.ones(len(ell))
    else:
        AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.
        AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1.
        AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.
        AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.
        AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.
        AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280


    ## make an array of noise curves for P
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])
    N_ell_P_SA[:, 0:2] = 0
    ####################################################################
    return(ell, N_ell_P_SA)

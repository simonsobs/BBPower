from __future__ import print_function
import numpy as np

def so_V3_LA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimates these for you
    return(np.array([27,39,93,145,225,280]))

def so_V3_LA_beams():
    ## returns the LAC beams in arcminutes
    beam_LAC_27 = 7.4
    beam_LAC_39 = 5.1
    beam_LAC_93 = 2.2
    beam_LAC_145 = 1.4
    beam_LAC_225 =1.0
    beam_LAC_280 = 0.9
    return(np.array([beam_LAC_27,beam_LAC_39,beam_LAC_93,beam_LAC_145,beam_LAC_225,beam_LAC_280]))

def so_V3_LA_noise(sensitivity_mode,f_sky,ell_max,delta_ell=1,beam_corrected=False):
    ## retuns noise curves, including the impact of the beam for the SO large aperature telescopes
    # sensitivity_mode
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the compuatioan of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## LARGE APERATURE
    # configuraiton
    NTubes_LF  = 1.
    NTubes_MF  = 4.
    NTubes_UHF = 2.
    # sensitivity
    S_LA_27  = np.array([61,48,35]) * np.sqrt(1./NTubes_LF)  ## converting these to per tube sensitivities
    S_LA_39  = np.array([30,24,18]) * np.sqrt(1./NTubes_LF)
    S_LA_93  = np.array([6.5,5.4,3.9]) * np.sqrt(4./NTubes_MF)
    S_LA_145 = np.array([8.1,6.7,4.2]) * np.sqrt(4./NTubes_MF)
    S_LA_225 = np.array([17,15,10]) * np.sqrt(2./NTubes_UHF)
    S_LA_280 = np.array([42,36,25]) * np.sqrt(2./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_LA_27 = 700.
    f_knee_pol_LA_39 = 700.
    f_knee_pol_LA_93 = 700.
    f_knee_pol_LA_145 = 700.
    f_knee_pol_LA_225 = 700.
    f_knee_pol_LA_280 = 700.
    alpha_pol =-1.4
    # atmospheric 1/f temp from matthew's model
    C_27 =    200.
    C_39 =      7.7
    C_93 =   1800.
    C_145 = 12000.
    C_225 = 68000.
    C_280 =124000.
    alpha_temp = -3.5
    
    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    t = t* 0.85  ## a kluge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky areas in Steridians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_LA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_LA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_LA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)
    
    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels= np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ## calculate the astmospheric contribution for T (based on Matthews model)
    ell_pivot = 1000.
    AN_T_27  = C_27  * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_LF)
    AN_T_39  = C_39  * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_LF)
    AN_T_93  = C_93  * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_MF)
    AN_T_145 = C_145 * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_MF)
    AN_T_225 = C_225 * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_UHF)
    AN_T_280 = C_280 * (ell/ell_pivot)**alpha_temp * A_SR / t / np.sqrt(NTubes_UHF)
    
    ## calculate N(ell)
    N_ell_T_27   = W_T_27**2. * A_SR + AN_T_27
    N_ell_T_39   = W_T_39**2. * A_SR + AN_T_39
    N_ell_T_93   = W_T_93**2. * A_SR + AN_T_93
    N_ell_T_145  = W_T_145**2.* A_SR + AN_T_145
    N_ell_T_225  = W_T_225**2.* A_SR + AN_T_225
    N_ell_T_280  = W_T_280**2.* A_SR + AN_T_280
    
    ## include the imapct of the beam
    LA_beams = so_V3_LA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
    ## lac beams as a sigma expressed in radians
    if beam_corrected :
        N_ell_T_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2 )
        N_ell_T_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2 )
        N_ell_T_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2 )
        N_ell_T_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2 )
        N_ell_T_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2 )
        N_ell_T_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2 )
    
    ## make an array of nosie curves for T
    N_ell_T_LA = np.array([N_ell_T_27,N_ell_T_39,N_ell_T_93,N_ell_T_145,N_ell_T_225,N_ell_T_280])
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the astmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_LA_27 )**alpha_pol + 1.
    AN_P_39  = (ell / f_knee_pol_LA_39 )**alpha_pol + 1.
    AN_P_93  = (ell / f_knee_pol_LA_93 )**alpha_pol + 1.
    AN_P_145 = (ell / f_knee_pol_LA_145)**alpha_pol + 1.
    AN_P_225 = (ell / f_knee_pol_LA_225)**alpha_pol + 1.
    AN_P_280 = (ell / f_knee_pol_LA_280)**alpha_pol + 1.
    
    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280
    
    ## include the imapct of the beam
    if beam_corrected :
        N_ell_P_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2 )
        N_ell_P_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2 )
        N_ell_P_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2 )
        N_ell_P_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2 )
        N_ell_P_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2 )
        N_ell_P_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2 )
    
    ## make an array of nosie curves for T
    N_ell_P_LA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])
    
    ####################################################################
    return(ell, N_ell_T_LA,N_ell_P_LA,Map_white_noise_levels)

def so_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimates these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def so_V3_SA_beams():
    ## returns the LAC beams in arcminutes
    beam_SAC_27 = 91.
    beam_SAC_39 = 63.
    beam_SAC_93 = 30.
    beam_SAC_145 = 17.
    beam_SAC_225 = 11.
    beam_SAC_280 = 9.
    return(np.array([beam_SAC_27,beam_SAC_39,beam_SAC_93,beam_SAC_145,beam_SAC_225,beam_SAC_280]))

def so_V3_SA_noise(sensitivity_mode,one_over_f_mode,SAC_yrs_LF,f_sky,ell_max,delta_ell=1,
                   beam_corrected=False,remove_kluge=False):
    ## retuns noise curves, including the impact of the beam for the SO small aperature telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    #     2: none
    # SAC_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAC
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the compuatioan of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERATURE
    ## LARGE APERATURE
    # configuraiton
    if (SAC_yrs_LF > 0):
        NTubes_LF  = SAC_yrs_LF/5. + 1e-6  ## reguarlized in case zero years is called
        NTubes_MF  = 2 - SAC_yrs_LF/5.
    else:
        NTubes_LF  = -SAC_yrs_LF/5. + 1e-6  ## reguarlized in case zero years is called
        NTubes_MF  = 2
    NTubes_UHF = 1.
    # sensitivity
    S_SA_27  = np.array([32,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([17,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([4.6,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([5.5,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([11,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([26,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_SA_27  = np.array([ 30.,15.,0.1])
    f_knee_pol_SA_39  = np.array([ 30.,15.,0.1])  ## from QUIET
    f_knee_pol_SA_93  = np.array([ 50.,25.,0.1])
    f_knee_pol_SA_145 = np.array([ 50.,25.,0.1])  ## from ABS, improving possible by scanning faster
    f_knee_pol_SA_225 = np.array([ 70.,35.,0.1])
    f_knee_pol_SA_280 = np.array([100.,40.,0.1])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])  ## roughly consistent with Yuji's table, but ectrapolated
    #alpha_pol =np.array([-2.4,-2.4,-2.4,-2.4,-2.4,-2.4])  ## roughly consistent with Yuji's table, but ectrapolated
    
    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    if remove_kluge==False :
        t = t* 0.85  ## a kluge for the noise non-uniformity of the map edges
        #print("when generating relizations from a hits map, the total integration time should be 1/0.85 longer")
        #print("since we should remove a cluge for map non-uniformity since this is included correcly in a hits map")
    A_SR = 4 * np.pi * f_sky  ## sky areas in Steridians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
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
    Map_white_noise_levels = np.sqrt(2)*np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the astmospheric contribution for P
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
    
    ## include the imapct of the beam
    SA_beams = so_V3_SA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
    ## lac beams as a sigma expressed in radians
    if beam_corrected :
        N_ell_P_27  *= np.exp( ell*(ell+1)* SA_beams[0]**2 )
        N_ell_P_39  *= np.exp( ell*(ell+1)* SA_beams[1]**2 )
        N_ell_P_93  *= np.exp( ell*(ell+1)* SA_beams[2]**2 )
        N_ell_P_145 *= np.exp( ell*(ell+1)* SA_beams[3]**2 )
        N_ell_P_225 *= np.exp( ell*(ell+1)* SA_beams[4]**2 )
        N_ell_P_280 *= np.exp( ell*(ell+1)* SA_beams[5]**2 )
    
    ## make an array of nosie curves for T
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])
    
    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)

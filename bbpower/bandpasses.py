import numpy as np


class Bandpass(object):
    #
    # Attributes:
    #   number: refers to the current object's band number (e.g. band 1 has number = 1) 
    #   nu: a list of frequencies within the sampling window; used to divide the window into 'bins' 
    #   bnu_dnu: the product of a bin width and the telescope's bandpass in a give bin. 
    #       useful since measured intensity within a bin can be approximated by the average emmitted intensity in the bin times bnu_dnu times nu^2
    #   nu_mean: the mean frequency of CMB radiation observed (weighted by intensity)
    #   cmb_norm: normalisation constant, preportional to the intensity of CMB radiation measured 
    # Additional values from config -> systematics -> bandpasses -> bandpass{self.number} -> parameters (default is false / none if location is empty)
    #   do_shift
    #   name_shift
    #   do_gain
    #   name_gain
    #   do_dphi1
    #   name_dphi1
    def __init__(self, nu, dnu, bnu, bp_number, config, phi_nu=None):
        # 
        # Input:
        #   nu: a list of frequencies within the sampling window at which the telescope's sensitivity is known
        #   dnu: list of 'width' of nu around each frequency in nu
        #       calcualated as dnu[0] = nu[1]-nu[0], dnu[-1] = nu[-1]-nu[-2], dnu[i] = 0.5*(nu[i+1]-nu[i-1]) for all other i
        #   bnu: list of sensitivities of the telescope at each frequency in nu 
        #   bp_number: band number, equal to 1 for the first band, 2 for the second ...
        #   config: the configuration file
        #   is_complex: BOOL, true if config -> systematics -> bandpasses -> bandpass_{self.number} -> phase_nu is none-empty
        #
        # No output
        self.number = bp_number
        self.nu = nu
        self.bnu_dnu = bnu * dnu
        cmbs = self.sed_CMB_RJ(self.nu)
        self.nu_mean = (np.sum(cmbs * self.bnu_dnu * nu**3) /
                        np.sum(cmbs * self.bnu_dnu * nu**2))
        self.cmb_norm = np.sum(cmbs * self.bnu_dnu * nu**2)
        field = 'bandpass_%d' % bp_number

        # Get frequency-dependent angle if necessary
        try:
            fname = config['systematics']['bandpasses'][field]['phase_nu']
        except KeyError:
            fname = None
            self.is_complex = False
        if fname:
            from scipy.interpolate import interp1d
            nu_phi, phi = np.loadtxt(fname, unpack=True)
            phif = interp1d(nu_phi, np.radians(phi),
                            bounds_error=False, fill_value=0)
            phi_arr = phif(self.nu)
            phase = np.cos(2*phi_arr) + 1j * np.sin(2*phi_arr)
            self.bnu_dnu = self.bnu_dnu * phase
            self.is_complex = True

        # Checking if we'll be sampling over bandpass systematics
        self.do_shift = False
        self.name_shift = None
        self.do_gain = False
        self.name_gain = None
        self.do_angle = False
        self.name_angle = None
        self.do_dphi1 = False
        self.name_dphi1 = None
        try:
            d = config['systematics']['bandpasses'][field]['parameters']
        except KeyError:
            d = {}
        for n, p in d.items():
            if p[0] == 'shift':
                self.do_shift = True
                self.name_shift = n
            if p[0] == 'gain':
                self.do_gain = True
                self.name_gain = n
            if p[0] == 'angle':
                self.do_angle = True
                self.name_angle = n
            if p[0] == 'dphi1':
                self.do_dphi1 = True
                self.is_complex = True
                self.name_dphi1 = n
        return

    def sed_CMB_RJ(self, nu):
        # Input:
        #   nu: an array of frequencies
        #   
        # Output:
        #  conv_sed: an array of SED(nu) measured in RJ temperature units
        #   
        x = 0.01760867023799751*nu
        ex = np.exp(x)
        return ex*(x/(ex-1))**2

    def convolve_sed(self, sed, params):
        #
        # Finds the mean value of the sed observed in the current band, taking into account the finite width of the band, gain factors, phase shifts and frequency shifts defined in the config file
        #
        # Input:
        #   sed: the sed of interest
        #   params: a dictionary of {NAME: VAL} pairs for each parameter in the model. The mean sed is calculated with the parameters taking these values. 
        #
        # Output:
        #   conv_sed: the mean value of sed measured in the ban
        #   rot_mat: equal to None if do_dphi = False. Equal to a rotation matrix if not. 
        dnu = 0.
        dphi1_phase = 1.
        if self.do_shift:
            dnu = params[self.name_shift] * self.nu_mean

        if self.do_dphi1:
            dphi1 = params[self.name_dphi1]
            normed_dphi1 = dphi1 * np.pi / 180. * (self.nu - self.nu_mean) / self.nu_mean
            dphi1_phase = np.cos(2.*normed_dphi1) + 1j * np.sin(2.*normed_dphi1)

        nu_prime = self.nu + dnu
        # CMB sed
        if sed is None:
            sed = self.sed_CMB_RJ
        conv_sed = np.sum(sed(nu_prime) * self.bnu_dnu *
                          dphi1_phase * nu_prime**2) / self.cmb_norm

        if self.do_gain:
            conv_sed *= params[self.name_gain]

        if self.is_complex:
            mod = abs(conv_sed)
            cs = conv_sed.real/mod
            sn = conv_sed.imag/mod
            return mod, np.array([[cs, sn],
                                  [-sn, cs]])
        else:
            return conv_sed, None

    def get_rotation_matrix(self, params):
        if self.do_angle:
            phi = np.radians(params[self.name_angle])
            c = np.cos(2*phi)
            s = np.sin(2*phi)
            return np.array([[c, s],
                             [-s, c]])
        else:
            return None


def rotate_cells_mat(mat1, mat2, cls):
    if mat1 is not None:
        cls = np.einsum('ijk,lk', cls, mat1)
    if mat2 is not None:
        cls = np.einsum('jk,ikl', mat2, cls)
    return cls


def rotate_cells(bp1, bp2, cls, params):
    m1 = bp1.get_rotation_matrix(params)
    m2 = bp2.get_rotation_matrix(params)
    return rotate_cells_mat(m1, m2, cls)


def decorrelated_bpass(bpass1, bpass2, sed, params, decorr_delta):
    def convolved_freqs(bpass):
        dnu = 0.
        if bpass.do_shift:
            dnu = params[bpass.name_shift] * bpass.nu_mean
        nu_prime = bpass.nu + dnu
        bnu_prime = np.abs(bpass.bnu_dnu) * nu_prime**2
        bphi = bnu_prime * sed(nu_prime)
        return nu_prime, bphi

    nu_prime1, bphi1 = convolved_freqs(bpass1)
    nu_prime2, bphi2 = convolved_freqs(bpass2)
    nu1nu2 = np.outer(nu_prime1, 1./nu_prime2)
    decorr_exp = decorr_delta**(np.log(nu1nu2)**2)
    decorr_sed = np.einsum('i, ij, j', bphi1, decorr_exp, bphi2)
    decorr_sed *= 1./(bpass1.cmb_norm * bpass2.cmb_norm)

    if bpass1.do_gain:
        decorr_sed *= params[bpass1.name_gain]
    if bpass2.do_gain:
        decorr_sed *= params[bpass2.name_gain]
    return decorr_sed

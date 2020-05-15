import numpy as np

from fgbuster.component_model import CMB

class Bandpass(object):
    def __init__(self, nu, dnu, bnu, bp_number, config, phi_nu=None):
        self.number = bp_number
        self.nu = nu
        self.bnu_dnu = bnu * nu**2 * dnu
        self.nu_mean = np.sum(CMB('K_RJ').eval(self.nu) * self.bnu_dnu * nu) / \
                       np.sum(CMB('K_RJ').eval(self.nu) * self.bnu_dnu)
        field = 'bandpass_%d' % bp_number

        # Get frequency-dependent angle if necessary
        try:
            fname=config['systematics']['bandpasses'][field]['phase_nu']
        except KeyError:
            fname=None
            self.is_complex = False
        if fname:
            from scipy.interpolate import interp1d
            nu_phi,phi=np.loadtxt(fname,unpack=True)
            phif=interp1d(nu_phi, np.radians(phi),
                          bounds_error=False, fill_value=0)
            phi_arr=phif(self.nu)
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

        self.cmb_norm = 1./np.sum(CMB('K_RJ').eval(self.nu) * self.bnu_dnu)

    def convolve_sed(self, sed, params):
        dnu = 0.
        if self.do_shift:
            dnu = params[self.name_shift] * self.nu_mean

        conv_sed = np.sum(sed(self.nu + dnu) * self.bnu_dnu) * self.cmb_norm

        if self.do_gain:
            conv_sed *= params[self.name_gain]

        if self.is_complex:
            mod = abs(conv_sed)
            cs = conv_sed.real/mod
            sn = conv_sed.imag/mod
            return mod, np.array([[cs,sn],[-sn,cs]])
        else:
            return conv_sed, None

    def get_rotation_matrix(self, params):
        if self.do_angle:
            phi = params[self.name_angle]
            c=np.cos(2*phi)
            s=np.sin(2*phi)
            return np.array([[c,s],[-s,c]])
        else:
            return None

def rotate_cells_mat(mat1, mat2, cls):
    if mat1 is not None:
        cls=np.einsum('ijk,lk',cls,mat1)
    if mat2 is not None:
        cls=np.einsum('jk,ikl',mat2,cls)
    return cls

def rotate_cells(bp1, bp2, cls, params):
    m1=bp1.get_rotation_matrix(params)
    m2=bp2.get_rotation_matrix(params)
    return rotate_cells_mat(m1,m2,cls)

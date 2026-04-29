from __future__ import annotations

from collections.abc import Callable

import numpy as np


class Bandpass:
    """Frequency bandpass with optional instrumental systematics.

    Represents a single frequency channel bandpass and supports convolution
    with spectral energy distributions (SEDs). Can model systematics including
    frequency shift, gain calibration, polarization angle rotation, and
    frequency-dependent birefringence (dphi1). Complex bandpasses arise from
    HWP-like phase effects or dphi1 systematics.

    Parameters
    ----------
    nu : array_like
        Frequency array in GHz.
    dnu : array_like
        Frequency bin widths.
    bnu : array_like
        Bandpass transmission values.
    bp_number : int
        Bandpass identifier number, used to look up systematics config.
    config : dict
        Configuration dictionary. Systematics are read from
        ``config['systematics']['bandpasses'][f'bandpass_{bp_number}']``.

    Attributes
    ----------
    number : int
        Bandpass identifier number.
    nu : array_like
        Frequency array in GHz.
    bnu_dnu : array_like
        Product of transmission and bin width (possibly complex).
    nu_mean : float
        Transmission-weighted mean frequency.
    cmb_norm : float
        CMB SED normalization factor.
    is_complex : bool
        Whether the bandpass has complex (phase) information.
    """

    def __init__(
        self,
        nu: np.ndarray,
        dnu: np.ndarray,
        bnu: np.ndarray,
        bp_number: int,
        config: dict,
    ) -> None:
        self.number = bp_number
        self.nu = nu
        self.bnu_dnu = bnu * dnu
        cmbs = self.sed_CMB_RJ(self.nu)
        self.nu_mean = np.sum(cmbs * self.bnu_dnu * nu**3) / np.sum(
            cmbs * self.bnu_dnu * nu**2
        )
        self.cmb_norm = np.sum(cmbs * self.bnu_dnu * nu**2)
        field = f"bandpass_{bp_number}"

        # Get frequency-dependent angle if necessary
        try:
            fname = config["systematics"]["bandpasses"][field]["phase_nu"]
        except KeyError:
            fname = None
            self.is_complex = False
        if fname:
            from scipy.interpolate import interp1d

            nu_phi, phi = np.loadtxt(fname, unpack=True)
            phif = interp1d(nu_phi, np.radians(phi), bounds_error=False, fill_value=0)
            phi_arr = phif(self.nu)
            phase = np.cos(2 * phi_arr) + 1j * np.sin(2 * phi_arr)
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
            d = config["systematics"]["bandpasses"][field]["parameters"]
        except KeyError:
            d = {}
        for n, p in d.items():
            if p[0] == "shift":
                self.do_shift = True
                self.name_shift = n
            if p[0] == "gain":
                self.do_gain = True
                self.name_gain = n
            if p[0] == "angle":
                self.do_angle = True
                self.name_angle = n
            if p[0] == "dphi1":
                self.do_dphi1 = True
                self.is_complex = True
                self.name_dphi1 = n

    def sed_CMB_RJ(self, nu: np.ndarray) -> np.ndarray:
        """Compute the CMB spectral energy distribution in Rayleigh-Jeans units.

        Parameters
        ----------
        nu : array_like
            Frequencies in GHz.

        Returns
        -------
        array_like
            CMB SED evaluated at the given frequencies, in RJ temperature
            units (i.e., the conversion factor from CMB thermodynamic to RJ).
        """
        x = 0.01760867023799751 * nu
        ex = np.exp(x)
        return ex * (x / (ex - 1)) ** 2

    def convolve_sed(
        self, sed: Callable | None, params: dict
    ) -> tuple[float | complex, np.ndarray | None]:
        """Convolve an SED function with this bandpass.

        Applies frequency shift, gain, and dphi1 systematics if enabled.
        For complex bandpasses (HWP phase or dphi1), returns the amplitude
        and a 2x2 rotation matrix encoding the effective polarization angle.

        Parameters
        ----------
        sed : callable or None
            SED function ``sed(nu)`` returning the emission spectrum. If
            None, the CMB SED is used.
        params : dict
            Parameter dictionary containing systematic parameter values
            keyed by their configured names.

        Returns
        -------
        amplitude : float
            Bandpass-convolved SED amplitude, normalized to the CMB.
        rotation_matrix : ndarray or None
            A 2x2 rotation matrix if the bandpass is complex, otherwise
            None.
        """
        dnu = 0.0
        dphi1_phase = 1.0
        if self.do_shift:
            dnu = params[self.name_shift] * self.nu_mean

        if self.do_dphi1:
            dphi1 = params[self.name_dphi1]
            normed_dphi1 = (
                dphi1 * np.pi / 180.0 * (self.nu - self.nu_mean) / self.nu_mean
            )
            dphi1_phase = np.cos(2.0 * normed_dphi1) + 1j * np.sin(2.0 * normed_dphi1)

        nu_prime = self.nu + dnu
        # CMB sed
        if sed is None:
            sed = self.sed_CMB_RJ
        conv_sed = (
            np.sum(sed(nu_prime) * self.bnu_dnu * dphi1_phase * nu_prime**2)
            / self.cmb_norm
        )

        if self.do_gain:
            conv_sed *= params[self.name_gain]

        if self.is_complex:
            mod = abs(conv_sed)
            cs = conv_sed.real / mod
            sn = conv_sed.imag / mod
            return mod, np.array([[cs, sn], [-sn, cs]])
        else:
            return conv_sed, None

    def get_rotation_matrix(self, params: dict) -> np.ndarray | None:
        """Return a 2x2 polarization rotation matrix.

        Constructs a rotation matrix for polarization angle systematics.
        The rotation is by twice the angle parameter (standard for
        Stokes Q/U).

        Parameters
        ----------
        params : dict
            Parameter dictionary containing the angle systematic value
            (in degrees) if enabled.

        Returns
        -------
        ndarray or None
            A 2x2 rotation matrix ``[[cos2a, sin2a], [-sin2a, cos2a]]``
            if angle systematics are enabled, otherwise None.
        """
        if self.do_angle:
            phi = np.radians(params[self.name_angle])
            c = np.cos(2 * phi)
            s = np.sin(2 * phi)
            return np.array([[c, s], [-s, c]])
        else:
            return None


def rotate_cells_mat(
    mat1: np.ndarray | None, mat2: np.ndarray | None, cls: np.ndarray
) -> np.ndarray:
    """Apply rotation matrices to power spectrum arrays.

    Rotates the power spectra ``cls`` by the given 2x2 matrices using
    Einstein summation. Either or both matrices may be None (no rotation).

    Parameters
    ----------
    mat1 : ndarray or None
        2x2 rotation matrix for the first bandpass.
    mat2 : ndarray or None
        2x2 rotation matrix for the second bandpass.
    cls : ndarray
        Power spectrum array with shape ``(n_pol, n_pol, n_ell)`` or
        compatible.

    Returns
    -------
    ndarray
        Rotated power spectrum array.
    """
    if mat1 is not None:
        cls = np.einsum("ijk,lk", cls, mat1)
    if mat2 is not None:
        cls = np.einsum("jk,ikl", mat2, cls)
    return cls


def rotate_cells(
    bp1: Bandpass, bp2: Bandpass, cls: np.ndarray, params: dict
) -> np.ndarray:
    """Rotate power spectra using polarization angle systematics.

    Convenience wrapper that obtains rotation matrices from two Bandpass
    objects and applies them to the power spectrum array.

    Parameters
    ----------
    bp1 : Bandpass
        First bandpass object.
    bp2 : Bandpass
        Second bandpass object.
    cls : ndarray
        Power spectrum array to rotate.
    params : dict
        Parameter dictionary passed to ``Bandpass.get_rotation_matrix``.

    Returns
    -------
    ndarray
        Rotated power spectrum array.
    """
    m1 = bp1.get_rotation_matrix(params)
    m2 = bp2.get_rotation_matrix(params)
    return rotate_cells_mat(m1, m2, cls)


def decorrelated_bpass(
    bpass1: Bandpass, bpass2: Bandpass, sed: Callable, params: dict, decorr_delta: float
) -> float:
    """Compute the decorrelated bandpass-convolved SED for two bandpasses.

    Models frequency decorrelation between two bandpasses using the factor
    ``decorr_delta ** (log(nu1/nu2))^2``, which suppresses correlations
    between widely separated frequencies.

    Parameters
    ----------
    bpass1 : Bandpass
        First bandpass object.
    bpass2 : Bandpass
        Second bandpass object.
    sed : callable
        SED function ``sed(nu)`` evaluated at the shifted frequencies.
    params : dict
        Parameter dictionary containing systematic parameter values.
    decorr_delta : float
        Decorrelation parameter. Values less than 1 produce stronger
        decorrelation for larger frequency separations.

    Returns
    -------
    float
        Decorrelated cross-bandpass SED amplitude, normalized to CMB
        and including any gain systematics.
    """

    def convolved_freqs(bpass):
        dnu = 0.0
        if bpass.do_shift:
            dnu = params[bpass.name_shift] * bpass.nu_mean
        nu_prime = bpass.nu + dnu
        bnu_prime = np.abs(bpass.bnu_dnu) * nu_prime**2
        bphi = bnu_prime * sed(nu_prime)
        return nu_prime, bphi

    nu_prime1, bphi1 = convolved_freqs(bpass1)
    nu_prime2, bphi2 = convolved_freqs(bpass2)
    nu1nu2 = np.outer(nu_prime1, 1.0 / nu_prime2)
    decorr_exp = decorr_delta ** (np.log(nu1nu2) ** 2)
    decorr_sed = np.einsum("i, ij, j", bphi1, decorr_exp, bphi2)
    decorr_sed *= 1.0 / (bpass1.cmb_norm * bpass2.cmb_norm)

    if bpass1.do_gain:
        decorr_sed *= params[bpass1.name_gain]
    if bpass2.do_gain:
        decorr_sed *= params[bpass2.name_gain]
    return decorr_sed

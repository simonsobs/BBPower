from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from bbpipe import PipelineStage
from .types import FitsFile, YamlFile, DirFile
from .fg_model import FGModel
from .param_manager import ParameterManager
from .bandpasses import Bandpass, rotate_cells, rotate_cells_mat, decorrelated_bpass
from .likelihood import Likelihood
from . import samplers
import sacc


class BBCompSep(PipelineStage):
    """
    Component separation stage for harmonic-domain foreground cleaning.

    Performs multi-frequency component separation (e.g. BICEP-style) by
    fitting a parametric foreground and CMB model to cross-frequency power
    spectra. The foreground/CMB model and its free parameters are defined
    in the pipeline config file. Sampling is dispatched to the samplers
    module, which supports multiple backends (emcee, polychord, scipy, etc.).
    """

    name = "BBCompSep"
    inputs = [
        ("cells_coadded", FitsFile),
        ("cells_noise", FitsFile),
        ("cells_fiducial", FitsFile),
        ("cells_coadded_cov", FitsFile),
    ]
    outputs = [("output_dir", DirFile), ("config_copy", YamlFile)]
    config_options = {
        "likelihood_type": "h&l",
        "n_iters": 32,
        "nwalkers": 16,
        "r_init": 1.0e-3,
        "sampler": "emcee",
        "bands": "all",
    }

    def setup_compsep(self) -> None:
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        if self.config["fg_model"].get("use_moments"):
            self.precompute_w3j()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.params = ParameterManager(self.config)
        self.likelihood = Likelihood(
            model_func=self.model,
            param_manager=self.params,
            bbdata=self.bbdata,
            bbnoise=self.bbnoise,
            invcov=self.invcov,
            matrix_to_vector=self.matrix_to_vector,
            use_handl=self.use_handl,
            bbfiducial=getattr(self, "bbfiducial", None),
        )

    def get_moments_lmax(self) -> int:
        """Return the maximum multipole for the moment expansion."""
        return self.config["fg_model"].get("moments_lmax", 384)

    def precompute_w3j(self) -> None:
        """Precompute Wigner 3-j symbols for the moment expansion.

        Populates ``self.big_w3j``, a 3-D array of squared Wigner 3-j
        coefficients indexed by (ell, ell1, ell2), used by the 1x1 and
        0x2 moment evaluations.
        """
        from pyshtools.utils import Wigner3j

        lmax = self.get_moments_lmax()
        ells_w3j = np.arange(0, lmax)
        w3j = np.zeros_like(ells_w3j, dtype=float)
        self.big_w3j = np.zeros((lmax, lmax, lmax))
        for ell1 in ells_w3j[1:]:
            for ell2 in ells_w3j[1:]:
                w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
                w3j_array = w3j_array[: ellmax - ellmin + 1]
                # make the w3j_array the same shape as the w3j
                if len(w3j_array) < len(ells_w3j):
                    reference = np.zeros(len(w3j))
                    reference[: w3j_array.shape[0]] = w3j_array
                    w3j_array = reference

                w3j_array = np.concatenate([w3j_array[-ellmin:], w3j_array[:-ellmin]])
                w3j_array = w3j_array[: len(ells_w3j)]
                w3j_array[:ellmin] = 0

                self.big_w3j[:, ell1, ell2] = w3j_array

        self.big_w3j = self.big_w3j**2

    def matrix_to_vector(self, mat: np.ndarray) -> np.ndarray:
        """Extract the upper-triangle elements of symmetric covariance matrices.

        Parameters
        ----------
        mat : array_like
            Array whose last two dimensions are (nmaps, nmaps).

        Returns
        -------
        vec : ndarray
            Upper-triangle elements along the last axis.
        """
        return mat[..., self.index_ut[0], self.index_ut[1]]

    def vector_to_matrix(self, vec: np.ndarray) -> np.ndarray:
        """Reconstruct a symmetric matrix from its upper-triangle elements.

        Parameters
        ----------
        vec : ndarray
            1-D or 2-D array of upper-triangle elements produced by
            ``matrix_to_vector``.

        Returns
        -------
        mat : ndarray
            Symmetric matrix (or batch of matrices) of shape
            ``(..., nmaps, nmaps)``.

        Raises
        ------
        ValueError
            If *vec* has more than 2 dimensions.
        """
        if vec.ndim == 1:
            mat = np.zeros([self.nmaps, self.nmaps])
            mat[self.index_ut] = vec
            mat = mat + mat.T - np.diag(mat.diagonal())
        elif vec.ndim == 2:
            mat = np.zeros([len(vec), self.nmaps, self.nmaps])
            mat[..., self.index_ut[0], self.index_ut[1]] = vec[..., :]
            for i, m in enumerate(mat):
                mat[i] = m + m.T - np.diag(m.diagonal())
        else:
            raise ValueError("Input vector can only be 1- or 2-D")
        return mat

    def _freq_pol_iterator(self) -> Iterator[tuple[int, int, int, int, int, int, int]]:
        """Yield index tuples for all unique frequency-polarization pairs.

        Yields
        ------
        b1, b2 : int
            Frequency-band indices.
        p1, p2 : int
            Polarization indices.
        m1, m2 : int
            Flattened map indices (pol + npol * band).
        icl : int
            Running cross-spectrum index.
        """
        icl = -1
        for b1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = p1 + self.npol * b1
                for b2 in range(b1, self.nfreqs):
                    if b1 == b2:
                        p2_r = range(p1, self.npol)
                    else:
                        p2_r = range(self.npol)
                    for p2 in p2_r:
                        m2 = p2 + self.npol * b2
                        icl += 1
                        yield b1, b2, p1, p2, m1, m2, icl

    def parse_sacc_file(self) -> None:
        """Read power spectra, bandpasses, and window functions from SACC files.

        Populates ``self.bbdata``, ``self.bbnoise``, ``self.bbcovar``,
        ``self.invcov``, ``self.bpss``, ``self.windows``, ``self.ell_b``,
        ``self.bpw_l``, and related attributes needed by the likelihood.
        """
        # Decide if you're using H&L
        self.use_handl = self.config["likelihood_type"] == "h&l"

        # Read data
        self.s = sacc.Sacc.load_fits(self.get_input("cells_coadded"))
        self.s_cov = sacc.Sacc.load_fits(self.get_input("cells_coadded_cov"))
        tr_comb = self.s.get_tracer_combinations()
        for tr1, tr2 in tr_comb:
            ind1 = self.s.indices(data_type="cl_bb", tracers=(tr1, tr2))
            ind2 = self.s_cov.indices(data_type="cl_bb", tracers=(tr1, tr2))
            assert np.all(ind1 == ind2), "Covariance sacc ordering is wrong"
        if self.use_handl:
            s_fid = sacc.Sacc.load_fits(self.get_input("cells_fiducial"))
            s_noi = sacc.Sacc.load_fits(self.get_input("cells_noise"))

        # Keep only desired correlations
        self.pols = self.config["pol_channels"]
        corr_all = ["cl_ee", "cl_eb", "cl_be", "cl_bb"]
        corr_keep = []
        for m1 in self.pols:
            for m2 in self.pols:
                clname = "cl_" + m1.lower() + m2.lower()
                corr_keep.append(clname)
        for c in corr_all:
            if c not in corr_keep:
                self.s.remove_selection(c)
                self.s_cov.remove_selection(c)
                if self.use_handl:
                    s_fid.remove_selection(c)
                    s_noi.remove_selection(c)

        # Scale cuts
        self.s.remove_selection(ell__gt=self.config["l_max"])
        self.s.remove_selection(ell__lt=self.config["l_min"])
        self.s_cov.remove_selection(ell__gt=self.config["l_max"])
        self.s_cov.remove_selection(ell__lt=self.config["l_min"])
        if self.use_handl:
            s_fid.remove_selection(ell__gt=self.config["l_max"])
            s_fid.remove_selection(ell__lt=self.config["l_min"])
            s_noi.remove_selection(ell__gt=self.config["l_max"])
            s_noi.remove_selection(ell__lt=self.config["l_min"])

        if self.config["bands"] == "all":
            tr_names = sorted(list(self.s.tracers.keys()))
        else:
            tr_names = self.config["bands"]
        self.nfreqs = len(tr_names)
        self.npol = len(self.pols)
        self.nmaps = self.nfreqs * self.npol
        self.index_ut = np.triu_indices(self.nmaps)
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.pol_order = dict(zip(self.pols, range(self.npol)))

        # Collect bandpasses
        self.bpss = []
        for i_t, tn in enumerate(tr_names):
            t = self.s.tracers[tn]
            nu = t.nu
            dnu = np.zeros_like(nu)
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]
            bnu = t.bandpass
            self.bpss.append(Bandpass(nu, dnu, bnu, i_t + 1, self.config))

        # Get ell sampling
        # Example power spectrum
        self.ell_b, _ = self.s.get_ell_cl(
            "cl_" + 2 * self.pols[0].lower(), tr_names[0], tr_names[0]
        )
        # Avoid l<2
        win0 = self.s.data[0]["window"]
        mask_w = win0.values > 1
        self.bpw_l = win0.values[mask_w]
        self.n_ell = len(self.bpw_l)
        self.n_bpws = len(self.ell_b)
        # D_ell factor
        self.dl2cl = 2 * np.pi / (self.bpw_l * (self.bpw_l + 1))
        self.windows = np.zeros([self.ncross, self.n_bpws, self.n_ell])

        # Get power spectra and covariances
        if self.config["bands"] == "all":
            if not (
                self.s_cov.covariance.covmat.shape[-1]
                == len(self.s.mean)
                == self.n_bpws * self.ncross
            ):
                raise ValueError("C_ell vector's size is wrong")

        v2d = np.zeros([self.n_bpws, self.ncross])
        if self.use_handl:
            v2d_noi = np.zeros([self.n_bpws, self.ncross])
            v2d_fid = np.zeros([self.n_bpws, self.ncross])
        cv2d = np.zeros([self.n_bpws, self.ncross, self.n_bpws, self.ncross])

        self.vector_indices = self.vector_to_matrix(
            np.arange(self.ncross, dtype=int)
        ).astype(int)
        self.indx = []

        # Parse into the right ordering
        itr1 = self._freq_pol_iterator()
        for b1, b2, p1, p2, m1, m2, ind_vec in itr1:
            t1 = tr_names[b1]
            t2 = tr_names[b2]
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cl_typ = f"cl_{pol1}{pol2}"
            ind_a = self.s.indices(cl_typ, (t1, t2))
            if len(ind_a) != self.n_bpws:
                raise ValueError(
                    "All power spectra need to be " "sampled at the same ells"
                )
            w = self.s.get_bandpower_windows(ind_a)
            self.windows[ind_vec, :, :] = w.weight[mask_w, :].T
            v2d[:, ind_vec] = np.array(self.s.mean[ind_a])
            if self.use_handl:
                _, v2d_noi[:, ind_vec] = s_noi.get_ell_cl(cl_typ, t1, t2)
                _, v2d_fid[:, ind_vec] = s_fid.get_ell_cl(cl_typ, t1, t2)
            itr2 = self._freq_pol_iterator()
            for b1b, b2b, p1b, p2b, m1b, m2b, ind_vecb in itr2:
                t1b = tr_names[b1b]
                t2b = tr_names[b2b]
                pol1b = self.pols[p1b].lower()
                pol2b = self.pols[p2b].lower()
                cl_typb = f"cl_{pol1b}{pol2b}"
                ind_b = self.s.indices(cl_typb, (t1b, t2b))
                cv2d[:, ind_vec, :, ind_vecb] = self.s_cov.covariance.covmat[ind_a][
                    :, ind_b
                ]

        # Store data
        self.bbdata = self.vector_to_matrix(v2d)
        if self.use_handl:
            self.bbnoise = self.vector_to_matrix(v2d_noi)
            self.bbfiducial = self.vector_to_matrix(v2d_fid)
        else:
            self.bbnoise = None
            self.bbfiducial = None
        self.bbcovar = cv2d.reshape(
            [self.n_bpws * self.ncross, self.n_bpws * self.ncross]
        )
        self.invcov = np.linalg.solve(self.bbcovar, np.identity(len(self.bbcovar)))

    def load_cmb(self) -> None:
        """Load CMB tensor, lensing, and scalar template spectra from files.

        Reads paths from ``self.config['cmb_model']['cmb_templates']`` and
        populates ``self.cmb_tens``, ``self.cmb_lens``, and ``self.cmb_scal``.
        """
        cmb_lensingfile = np.loadtxt(self.config["cmb_model"]["cmb_templates"][0])
        cmb_bbfile = np.loadtxt(self.config["cmb_model"]["cmb_templates"][1])

        self.cmb_ells = cmb_bbfile[:, 0]
        mask = (self.cmb_ells <= self.bpw_l.max()) & (self.cmb_ells > 1)
        self.cmb_ells = self.cmb_ells[mask]

        # TODO: this is a patch
        nell = len(self.cmb_ells)
        self.cmb_tens = np.zeros([self.npol, self.npol, nell])
        self.cmb_lens = np.zeros([self.npol, self.npol, nell])
        self.cmb_scal = np.zeros([self.npol, self.npol, nell])
        if "B" in self.config["pol_channels"]:
            ind = self.pol_order["B"]
            self.cmb_tens[ind, ind] = (
                cmb_bbfile[:, 3][mask] - cmb_lensingfile[:, 3][mask]
            )
            self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        if "E" in self.config["pol_channels"]:
            ind = self.pol_order["E"]
            self.cmb_tens[ind, ind] = (
                cmb_bbfile[:, 2][mask] - cmb_lensingfile[:, 2][mask]
            )
            self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]

    def integrate_seds(self, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Compute band-averaged foreground SED scaling factors.

        Convolves each foreground component SED with the instrumental
        bandpasses and, optionally, applies frequency decorrelation.

        Parameters
        ----------
        params : dict
            Current parameter values keyed by name.

        Returns
        -------
        fg_scaling : ndarray
            Shape ``(n_components, n_components, nfreqs, nfreqs)``
            frequency-frequency scaling matrix for each component pair.
        rot_matrices : ndarray
            Polarization rotation matrices from the bandpass convolution,
            shape ``(n_components, nfreqs, ...)``.
        """
        single_sed = np.zeros([self.fg_model.n_components, self.nfreqs])
        comp_scaling = np.zeros([self.fg_model.n_components, self.nfreqs, self.nfreqs])
        fg_scaling = np.zeros(
            [
                self.fg_model.n_components,
                self.fg_model.n_components,
                self.nfreqs,
                self.nfreqs,
            ]
        )
        rot_matrices = []

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp["cmb_n0_norm"]
            sed_params = [params[comp["names_sed_dict"][k]] for k in comp["sed"].params]
            rot_matrices.append([])

            def sed(nu):
                return comp["sed"].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b, rot = self.bpss[tn].convolve_sed(sed, params)
                single_sed[i_c, tn] = sed_b * units
                rot_matrices[i_c].append(rot)

            if comp["decorr"]:
                d_amp = params[comp["decorr_param_names"]["decorr_amp"]]
                d_nu01 = params[comp["decorr_param_names"]["decorr_nu01"]]
                d_nu02 = params[comp["decorr_param_names"]["decorr_nu02"]]
                decorr_delta = d_amp ** (1.0 / np.log(d_nu01 / d_nu02) ** 2)
                for f1 in range(self.nfreqs):
                    for f2 in range(f1, self.nfreqs):
                        sed_12 = decorrelated_bpass(
                            self.bpss[f1], self.bpss[f2], sed, params, decorr_delta
                        )
                        comp_scaling[i_c, f1, f2] = sed_12 * units * units
            else:
                comp_scaling[i_c] = np.outer(single_sed[i_c], single_sed[i_c])

        for i_c1, c_name1 in enumerate(self.fg_model.component_names):
            fg_scaling[i_c1, i_c1] = comp_scaling[i_c1]
            for c_name2, epsname in self.fg_model.components[c_name1][
                "names_x_dict"
            ].items():
                i_c2 = self.fg_model.component_order[c_name2]
                eps = params[epsname]
                fg_scaling[i_c1, i_c2] = eps * np.outer(
                    single_sed[i_c1], single_sed[i_c2]
                )
                fg_scaling[i_c2, i_c1] = eps * np.outer(
                    single_sed[i_c2], single_sed[i_c1]
                )
        return fg_scaling, np.array(rot_matrices)

    def evaluate_power_spectra(self, params: dict) -> np.ndarray:
        """Evaluate foreground angular power spectra from the config model.

        Parameters
        ----------
        params : dict
            Current parameter values keyed by name.

        Returns
        -------
        fg_pspectra : ndarray
            Shape ``(n_components, npol, npol, n_ell)`` foreground C_ell
            for each component, converted from D_ell to C_ell.
        """
        fg_pspectra = np.zeros(
            [self.fg_model.n_components, self.npol, self.npol, self.n_ell]
        )

        # Fill diagonal
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            for cl_comb, clfunc in comp["cl"].items():
                m1, m2 = cl_comb
                ip1 = self.pol_order[m1]
                ip2 = self.pol_order[m2]
                pspec_params = [
                    params[comp["names_cl_dict"][cl_comb][k]] for k in clfunc.params
                ]
                p_spec = clfunc.eval(self.bpw_l, *pspec_params) * self.dl2cl
                fg_pspectra[i_c, ip1, ip2] = p_spec
                if m1 != m2:
                    fg_pspectra[i_c, ip2, ip1] = p_spec

        return fg_pspectra

    def model(self, params: dict) -> np.ndarray:
        """Compute the full CMB + foreground model integrated over bandpasses and windows.

        Parameters
        ----------
        params : dict
            Named parameter dictionary (CMB and foreground parameters).

        Returns
        -------
        np.ndarray
            Model bandpowers with shape ``(n_ell, ncross_freq, ncross_freq)``.
        """
        # [npol,npol,nell]
        cmb_cell = (
            params["r_tensor"] * self.cmb_tens
            + params["A_lens"] * self.cmb_lens
            + self.cmb_scal
        ) * self.dl2cl
        # [nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes=[2, 0, 1])
        if self.config["cmb_model"].get("use_birefringence"):
            bi_angle = np.radians(params["birefringence"])
            c = np.cos(2 * bi_angle)
            s = np.sin(2 * bi_angle)
            bmat = np.array([[c, s], [-s, c]])
            cmb_cell = rotate_cells_mat(bmat, bmat, cmb_cell)

        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros(
            [self.nfreqs, self.nfreqs, self.n_ell, self.npol, self.npol]
        )
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        # SED scaling
        cmb_scaling = np.ones(self.nfreqs)
        cmb_rot = []
        for f1 in range(self.nfreqs):
            cs, crot = self.bpss[f1].convolve_sed(None, params)
            cmb_scaling[f1] = cs
            cmb_rot.append(crot)

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                cls = (
                    rotate_cells_mat(cmb_rot[f2], cmb_rot[f1], cmb_cell)
                    * cmb_scaling[f1]
                    * cmb_scaling[f2]
                )

                # Loop over component pairs
                for c1 in range(self.fg_model.n_components):
                    for c2 in range(self.fg_model.n_components):
                        mat1 = rot_m[c1, f1]
                        mat2 = rot_m[c2, f2]
                        if c1 == c2:
                            clrot = rotate_cells_mat(mat2, mat1, fg_cell[c1])
                        else:
                            # For cross component, enforcing EB term is zero.
                            cl_cross = np.zeros((self.n_ell, self.npol, self.npol))
                            for i in range(self.npol):
                                cl_cross[:, i, i] = np.sqrt(
                                    fg_cell[c1, :, i, i] * fg_cell[c2, :, i, i]
                                )
                            clrot = rotate_cells_mat(mat2, mat1, cl_cross)
                        cls += clrot * fg_scaling[c1, c2, f1, f2]
                cls_array_fg[f1, f2] = cls

        # Add moment terms if needed
        if self.config["fg_model"].get("use_moments"):
            # TODO: moments work with:
            # - B-only
            # - No polarization angle business
            # - Only power-law beta power spectra at l_pivot=80

            # Evaluate 1st/2nd order SED derivatives.
            # [nfreq, ncomp]
            fg_scaling_d1 = self.integrate_seds_der(params, order=1)
            fg_scaling_d2 = self.integrate_seds_der(params, order=2)

            # Compute 1x1 for each component
            # Compute 0x2 for each component (essentially this is sigma_beta)
            # Evaluate beta power spectra.
            lmax_mom = self.get_moments_lmax()
            # [ncomp, nell, npol, npol]
            cls_11 = np.zeros(
                [self.fg_model.n_components, self.n_ell, self.npol, self.npol]
            )
            # [ncomp, nell, npol, npol]
            cls_02 = np.zeros(
                [self.fg_model.n_components, self.n_ell, self.npol, self.npol]
            )
            for i_c, c_name in enumerate(self.fg_model.component_names):
                comp = self.fg_model.components[c_name]
                gamma = params[comp["names_moments_dict"]["gamma_beta"]]
                amp = params[comp["names_moments_dict"]["amp_beta"]] * 1e-6
                cl_betas = self.bcls(lmax=lmax_mom, gamma=gamma, amp=amp)
                cl_cc = fg_cell[i_c, :]
                # cls_1x1 = 0
                cls_1x1 = self.evaluate_1x1(
                    params, lmax=lmax_mom, cls_cc=cl_cc, cls_bb=cl_betas
                )
                cls_11[i_c, :lmax_mom, :, :] = cls_1x1
                # cls_0x2 = 0
                cls_0x2 = self.evaluate_0x2(
                    params, lmax=lmax_mom, cls_cc=cl_cc, cls_bb=cl_betas
                )
                cls_02[i_c, :lmax_mom, :, :] = cls_0x2

            # Add components scaled in frequency
            for f1 in range(self.nfreqs):
                # Note that we only need to fill in half of the frequencies
                for f2 in range(f1, self.nfreqs):
                    cls = np.zeros([self.n_ell, self.npol, self.npol])
                    for c1 in range(self.fg_model.n_components):
                        cls += (
                            fg_scaling_d1[f1, c1] * fg_scaling_d1[f2, c1] * cls_11[c1]
                        )
                        cls += (
                            0.5
                            * (
                                fg_scaling_d2[f1, c1]
                                * (fg_scaling[c1, c1, f2, f2]) ** 0.5
                                + fg_scaling_d2[f2, c1]
                                * (fg_scaling[c1, c1, f1, f1]) ** 0.5
                            )
                            * cls_02[c1]
                        )
                    cls_array_fg[f1, f2] += cls

        # Window convolution
        cls_array_list = np.zeros(
            [self.n_bpws, self.nfreqs, self.npol, self.nfreqs, self.npol]
        )
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1 * self.npol + p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2 * self.npol + p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :, p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        # Polarization angle rotation
        for f1 in range(self.nfreqs):
            for f2 in range(self.nfreqs):
                cls_array_list[:, f1, :, f2, :] = rotate_cells(
                    self.bpss[f2],
                    self.bpss[f1],
                    cls_array_list[:, f1, :, f2, :],
                    params,
                )

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def bcls(self, lmax: int, gamma: float, amp: float) -> np.ndarray:
        """Compute a power-law beta power spectrum for the moment expansion.

        Parameters
        ----------
        lmax : int
            Maximum multipole.
        gamma : float
            Power-law tilt (pivot at ell = 80).
        amp : float
            Amplitude of the beta spectrum.

        Returns
        -------
        bcls : ndarray
            Beta power spectrum of length *lmax*.
        """
        ls = np.arange(lmax)
        bcls = np.zeros(len(ls))
        bcls[2:] = (ls[2:] / 80.0) ** gamma
        return bcls * amp

    def integrate_seds_der(self, params: dict, order: int = 1) -> np.ndarray:
        """Compute band-averaged SED derivatives for the moment expansion.

        Parameters
        ----------
        params : dict
            Named parameter dictionary.
        order : int
            Derivative order (1 or 2).

        Returns
        -------
        np.ndarray
            SED derivative matrix of shape ``(nfreqs, n_components)``.
        """
        fg_scaling_der = np.zeros([self.fg_model.n_components, self.nfreqs])

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp["cmb_n0_norm"]
            sed_params = [params[comp["names_sed_dict"][k]] for k in comp["sed"].params]

            # Set SED function with scaling beta
            def sed_der(nu):
                nu0 = params[comp["names_sed_dict"]["nu0"]]
                x = np.log(nu / nu0)
                # This is only valid for spectral indices
                return x**order * comp["sed"].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b = self.bpss[tn].convolve_sed(sed_der, params)[0]
                fg_scaling_der[i_c, tn] = sed_b * units

        return fg_scaling_der.T

    def evaluate_1x1(
        self, params: dict, lmax: int, cls_cc: np.ndarray, cls_bb: np.ndarray
    ) -> np.ndarray:
        """Evaluate the first-order (1x1) moment expansion correction.

        Parameters
        ----------
        params : dict
            Named parameter dictionary.
        lmax : int
            Maximum multipole for the expansion.
        cls_cc : np.ndarray
            Cross-component power spectra.
        cls_bb : np.ndarray
            Beta auto-spectrum (spectral index variance).

        Returns
        -------
        np.ndarray
            1x1 moment correction term.
        """

        ls = np.arange(lmax)
        v_left = (2 * ls + 1)[:, None, None] * cls_cc[:lmax, :, :]
        v_right = (2 * ls + 1) * cls_bb[:lmax]

        mat = self.big_w3j
        v_left = np.transpose(v_left, axes=[1, 0, 2])
        # Contract the Wigner-3j tensor with the beta spectrum first; this is
        # noticeably faster than a pair of generic matrix multiplies here.
        tmp_moment = np.einsum("...j,j", mat, v_right, optimize="greedy")
        moment1x1 = np.dot(tmp_moment, v_left) / (4 * np.pi)
        return moment1x1

    def evaluate_0x2(
        self, params: dict, lmax: int, cls_cc: np.ndarray, cls_bb: np.ndarray
    ) -> np.ndarray:
        """Evaluate the zeroth-by-second-order (0x2) moment correction.

        Assumes a power-law spectral index field.

        Parameters
        ----------
        params : dict
            Named parameter dictionary.
        lmax : int
            Maximum multipole for the expansion.
        cls_cc : np.ndarray
            Cross-component power spectra.
        cls_bb : np.ndarray
            Beta auto-spectrum (spectral index variance).

        Returns
        -------
        np.ndarray
            0x2 moment correction term.
        """
        ls = np.arange(lmax)
        prefac = np.sum((2 * ls + 1) * cls_bb) / (4 * np.pi)
        return cls_cc[:lmax] * prefac

    def run(self) -> None:
        """Execute the component-separation pipeline stage.

        Copies the config file to the output directory, initialises the
        data and models via ``setup_compsep``, then dispatches to the
        configured sampler.
        """
        from shutil import copyfile

        copyfile(self.get_input("config"), self.get_output("config_copy"))
        self.setup_compsep()

        sampler_name = self.config.get("sampler", "emcee")
        output_dir = self.get_output("output_dir")

        if sampler_name == "predicted_spectra":
            samplers.run_predicted_spectra(
                self.likelihood, self, self.config, output_dir
            )
        elif sampler_name in samplers.SAMPLERS:
            samplers.SAMPLERS[sampler_name](self.likelihood, self.config, output_dir)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name!r}")


if __name__ == "__main__":
    cls = PipelineStage.main()

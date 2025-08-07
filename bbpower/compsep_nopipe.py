import numpy as np
import os
import argparse
import yaml
import time
import sacc  # noqa

import bbpower.mpi_utils as mpi
from bbpower.fg_model import FGModel
from bbpower.param_manager import ParameterManager
from bbpower.bandpasses import (Bandpass, rotate_cells, rotate_cells_mat,
                                decorrelated_bpass)

def _yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


class BBCompSep(object):
    # TODO: FIX the analytical derivatives wrt beta
    """
    Component separation stage
    This stage does harmonic domain foreground cleaning (e.g. BICEP).
    The foreground model parameters are defined in the config.yml file.
    """

    # Attributes:
    # 
    #   config: DICT - contains the info from the configuration file taht is stored within "global" or "BBCompSep"
    #   config_fname: ???
    #   use_handl: BOOL - true if we are using handl
    #   s: saac data of cross-split power spectra (i.e. the data we are analysing) with only the desired frequency channels
    #   s_cov: saac data of file with power spectrum covariance with only the desired frequency channels and desired polarization correlations within the range [l_min,l_max]
    #   s_fid: if use_handl is true, this is the saac data of the fiducial power spectra  and desired polarization correlations within the range [l_min,l_max]
    #   s_noi: if use_handl is true, this is the saac data of the noise power spectra at the desired polarization correlations within the range [l_min,l_max]
    #   pols: list of polarization channels to be considered
    #   nfreq: INT - the number of frequencies that we are considering
    #   npol: INT - the number of polerizations that we are considering
    #   nmap: INT - nfreq * npol, the number of maps that need to be considered
    #   index_ut: TUPLE of ndarrays. Contains the indecies of the upper triangle of a nmap by nmap matrix
    #   ncross: the number of (frequency1, polarisation1), (frequency2, polarization2) pairs that we will be considering
    #   pol_order: DICT of {POLARIZATION:INDEX} where index describes the order polarizations appear within the files
    #   bpss: list of Bypasses object - contains info regarding bandpasses for each filter that we observed with
    #   ell_b: list of ell that have been sampled
    #   bpw_l:
    #   n_ell: the ell values (>=2) that we have data for
    #   n_bpws: the ell values that we are actually sampling for
    #   dl2cl: array of conversion factors to convert from dl to cl for each ell value in npw_l
    #   vector_indices - a nmap x nmap symetric matrix with elements in the upper triangle labelled in ascending order, first by row then column:
    #       e.g. if nmap = 4, upper triangle is labelled:
    #               0  1  2  3
    #                  4  5  6
    #                     7  8
    #                        9
    #       so vector_indicies is the symetric matrix with those elements in the upper triangle:
    #              [ [  0,  1,  2,  3],
    #                [  1,  4,  5,  6],
    #                [  2,  5,  7,  8],
    #                [  3,  6,  8,  9] ]
    #   bbdata: 3dmatrix containing datapoints. First index is arranged in order of ell. Second index is arranged by frequency1 and then polarization1
    #           (e.g. if we had two frequency bands b1 and b2 and two polarizations E and B, the columns are b1E b1B b2E b2B) and then thrid index is 
    #           arranged by freq2 and then polarization2. 
    #   bbnoise: only defined if use_handl = true. 3d matrix containing noise data, arranged in the same order as bbdata
    #   bbfiducial: only defined if use_handl = true. 3d matrix containing fiducial values, arranged in the same order as bbdata
    #   bbcovar: the covariance matrix
    #   invcov: the inverse of the covariance matrix
    #   cmb_tens: array - contains the cmb_tens template
    #       indexes correspond to [polarization1][polarization2][ell] (e.g. if B is the first polarization mode, cmb_tens[0][0] lists the template's value for each ell that we consider, in order)
    #   cmb_lens: array - contains the cmb_lens template, in the same format as cmb_tens
    #   cmb_scal: array - contains the cmb_scal template, in the same format as cmb_tens
    #
    #   fgmodel: a FGModel object (class definition in fg_model.py) initialised with self.config
    #   params: a ParameterManager object (class definition in param_manager.py) initialised with self.config
    #

    def __init__(self, args):
        """
        Initialize from the command line arguments.

        Parameters
        ----------
        args : str
            Command line arguments.
        """

        # Load the configuration file
        config = _yaml_loader(args.config)
        self.config = config["global"] | config["BBCompSep"]
        setattr(self, "config_fname", getattr(args, "config"))
        setattr(self, "data", self.config["data"])

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        if self.config['fg_model'].get('use_moments'):
            self.precompute_w3j()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.params = ParameterManager(self.config)
        if self.use_handl:
            self.prepare_h_and_l()
        return

    def get_moments_lmax(self):
        return self.config['fg_model'].get('moments_lmax', 384)

    def precompute_w3j(self):
        from pyshtools.utils import Wigner3j

        lmax = self.get_moments_lmax()
        ells_w3j = np.arange(0, lmax)
        w3j = np.zeros_like(ells_w3j, dtype=float)
        self.big_w3j = np.zeros((lmax, lmax, lmax))
        for ell1 in ells_w3j[1:]:
            for ell2 in ells_w3j[1:]:
                w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
                w3j_array = w3j_array[:ellmax - ellmin + 1]
                # make the w3j_array the same shape as the w3j
                if len(w3j_array) < len(ells_w3j):
                    reference = np.zeros(len(w3j))
                    reference[:w3j_array.shape[0]] = w3j_array
                    w3j_array = reference

                w3j_array = np.concatenate([w3j_array[-ellmin:],
                                            w3j_array[:-ellmin]])
                w3j_array = w3j_array[:len(ells_w3j)]
                w3j_array[:ellmin] = 0

                self.big_w3j[:, ell1, ell2] = w3j_array

        self.big_w3j = self.big_w3j**2

    def matrix_to_vector(self, mat):
        return mat[..., self.index_ut[0], self.index_ut[1]]

    def vector_to_matrix(self, vec):
        # For a 1d input: 
        #   Converts a 1D array of length ncross into a symetric nmaps x nmaps matrix. The elements in vec are placed into the matrix in the same order as the integers in vector_indecies.
        #   (e.g. mat[0][0] = vec[0], mat[0][1] = mat[1][0] = vec[1] etc)
        # For a 2d input:
        #   carries out the above process piecewise to make a 3d array (i.e. mat[i] = vector_to_matrix(vec[i]))
        #
        # Input:
        #   vec: a 1d array of length ncross or a 2d array of length m x ncross
        # Output:
        #   mat: a nmap x nmap array (1d input) or m x nmap x nmap array (2d input) constructed as described above
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

    def _freq_pol_iterator(self):
        icl = -1
        map_sets = list(self.config["map_sets"])
        for b1 in range(len(map_sets)):
            for p1 in range(self.npol):
                m1 = p1 + self.npol * b1
                for b2 in range(b1, len(map_sets)):
                    if b1 == b2:
                        p2_r = range(p1, self.npol)
                    else:
                        p2_r = range(self.npol)
                    for p2 in p2_r:
                        m2 = p2 + self.npol * b2
                        icl += 1
                        yield map_sets[b1], map_sets[b2], p1, p2, m1, m2, icl

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra,
        bandpasses, and window functions.
        """
        from copy import deepcopy

        # Decide if you're using H&L
        self.use_handl = self.config['likelihood_type'] == 'h&l'

        # Read data
        cells_coadded = self.data["cells_coadded"].format(sim_id=self.sim_id)
        self.s = sacc.Sacc.load_fits(cells_coadded)
        self.s_cov = sacc.Sacc.load_fits(self.data["cells_coadded_cov"])
        if self.use_handl:
            s_fid = sacc.Sacc.load_fits(self.data["cells_fiducial"])
            s_noi = sacc.Sacc.load_fits(self.data["cells_noise"])

        # Keep only desired tracers
        tr_names = list(self.config['map_sets'].keys())
        tr_names_before = deepcopy(self.s.tracers)
        for tr in tr_names_before:
            if tr not in tr_names:
                self.s.remove_tracers([tr])
                self.s_cov.remove_tracers([tr])
        tr_comb = self.s.get_tracer_combinations()

        # Keep only desired correlations
        self.pols = self.config['pol_channels']
        corr_all = ['cl_00', 'cl_0e', 'cl_0b', 'cl_e0', 'cl_b0',
                    'cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        corr_keep = []
        for m1 in self.pols:
            for m2 in self.pols:
                clname = 'cl_' + m1.lower() + m2.lower()
                if "0" in clname:
                    continue
                corr_keep.append(clname)
        for c in corr_all:
            if c not in corr_keep:
                self.s.remove_selection(c)
                self.s_cov.remove_selection(c)
                if self.use_handl:
                    s_fid.remove_selection(c)
                    s_noi.remove_selection(c)

        # Scale cuts
        self.s.remove_selection(ell__gt=self.config['l_max'])
        self.s.remove_selection(ell__lt=self.config['l_min'])
        self.s_cov.remove_selection(ell__gt=self.config['l_max'])
        self.s_cov.remove_selection(ell__lt=self.config['l_min'])
        if self.use_handl:
            s_fid.remove_selection(ell__gt=self.config['l_max'])
            s_fid.remove_selection(ell__lt=self.config['l_min'])
            s_noi.remove_selection(ell__gt=self.config['l_max'])
            s_noi.remove_selection(ell__lt=self.config['l_min'])

        for tr1, tr2 in tr_comb:
            ind1 = self.s.indices(data_type='cl_bb', tracers=(tr1, tr2))
            ind2 = self.s_cov.indices(data_type='cl_bb', tracers=(tr1, tr2))
            assert np.all(ind1 == ind2), "Covariance sacc ordering is wrong"
            if self.use_handl:
                ind3 = self.s_fid.indices(data_type='cl_bb',
                                          tracers=(tr1, tr2))
                ind4 = self.s_noi.indices(data_type='cl_bb',
                                          tracers=(tr1, tr2))
                assert np.all(ind1 == ind3), "Fiducial sacc ordering is wrong"
                assert np.all(ind1 == ind4), "Noise sacc ordering is wrong"

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
            self.bpss.append(Bandpass(nu, dnu, bnu, i_t+1, self.config))

        # Get ell sampling
        # Example power spectrum
        self.ell_b, _ = self.s.get_ell_cl('cl_' + 2 * self.pols[0].lower(),
                                          tr_names[0], tr_names[0])
        # Avoid l<2
        win0 = self.s.data[0]['window']
        mask_w = win0.values > 1
        self.bpw_l = win0.values[mask_w]
        self.n_ell = len(self.bpw_l)
        self.n_bpws = len(self.ell_b)
        # D_ell factor
        self.dl2cl = 2 * np.pi / (self.bpw_l * (self.bpw_l + 1))
        self.windows = np.zeros([self.ncross, self.n_bpws, self.n_ell])

        # Get power spectra and covariances
        if not (self.s_cov.covariance.covmat.shape[-1] == len(self.s.mean)
                == self.n_bpws * self.ncross):
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
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cl_typ = f'cl_{pol1}{pol2}'
            ind_a = self.s.indices(cl_typ, (b1, b2))
            if len(ind_a) != self.n_bpws:
                raise ValueError("All power spectra need to be "
                                 "sampled at the same ells")
            w = self.s.get_bandpower_windows(ind_a)
            self.windows[ind_vec, :, :] = w.weight[mask_w, :].T
            v2d[:, ind_vec] = np.array(self.s.mean[ind_a])
            if self.use_handl:
                _, v2d_noi[:, ind_vec] = s_noi.get_ell_cl(cl_typ, b1, b2)
                _, v2d_fid[:, ind_vec] = s_fid.get_ell_cl(cl_typ, b1, b2)
            itr2 = self._freq_pol_iterator()
            for b1b, b2b, p1b, p2b, _, _, ind_vecb in itr2:
                pol1b = self.pols[p1b].lower()
                pol2b = self.pols[p2b].lower()
                cl_typb = f'cl_{pol1b}{pol2b}'
                ind_b = self.s.indices(cl_typb, (b1b, b2b))
                cv2d[:, ind_vec, :, ind_vecb] = self.s_cov.covariance.covmat[ind_a][:, ind_b]  # noqa

        # Store data
        self.bbdata = self.vector_to_matrix(v2d)
        if self.use_handl:
            self.bbnoise = self.vector_to_matrix(v2d_noi)
            self.bbfiducial = self.vector_to_matrix(v2d_fid)
        self.bbcovar = cv2d.reshape([self.n_bpws * self.ncross,
                                     self.n_bpws * self.ncross])
        self.invcov = np.linalg.solve(self.bbcovar,
                                      np.identity(len(self.bbcovar)))
        np.savez(self.output_dir + '/data_ell_cl_invcov.npz',
                 ell=self.ell_b, cl=self.bbdata, invcov=self.invcov)
        return

    def load_cmb(self):
        """
        Loads the CMB BB spectrum as defined in the config file.
        """
        cmb_lensingfile = np.loadtxt(
            self.config['cmb_model']['cmb_templates'][0]
        )
        cmb_bbfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][1])

        self.cmb_ells = cmb_bbfile[:, 0]
        mask = (self.cmb_ells <= self.bpw_l.max()) & (self.cmb_ells > 1)
        self.cmb_ells = self.cmb_ells[mask]

        # TODO: this is a patch
        nell = len(self.cmb_ells)
        self.cmb_tens = np.zeros([self.npol, self.npol, nell])
        self.cmb_lens = np.zeros([self.npol, self.npol, nell])
        self.cmb_scal = np.zeros([self.npol, self.npol, nell])
        if 'B' in self.config['pol_channels']:
            ind = self.pol_order['B']
            self.cmb_tens[ind, ind] = (cmb_bbfile[:, 3][mask] -
                                       cmb_lensingfile[:, 3][mask])
            self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        if 'E' in self.config['pol_channels']:
            ind = self.pol_order['E']
            self.cmb_tens[ind, ind] = (cmb_bbfile[:, 2][mask] -
                                       cmb_lensingfile[:, 2][mask])
            self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]
        return

    def integrate_seds(self, params):
        # 
        # Input:
        #   params: a dictionary of {NAME: VAL} pairs for the parameters. sed is evaluated at these values
        #
        # Output:
        #   fgscaling: a 4d array containing SED scalings for each component and frequency. Indecies represent (in order):
        #       [component1][component2][freq_band1][freq_band2]
        #   rot_matricies: a 1d array of rotation matricies for each frequency
        single_sed = np.zeros([self.fg_model.n_components,
                               self.nfreqs])
        comp_scaling = np.zeros([self.fg_model.n_components,
                                 self.nfreqs, self.nfreqs])
        fg_scaling = np.zeros([self.fg_model.n_components,
                               self.fg_model.n_components,
                               self.nfreqs, self.nfreqs])
        rot_matrices = []

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]
            rot_matrices.append([])

            def sed(nu):
                return comp['sed'].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b, rot = self.bpss[tn].convolve_sed(sed, params)
                single_sed[i_c, tn] = sed_b * units
                rot_matrices[i_c].append(rot)

            if comp['decorr']:
                d_amp = params[comp['decorr_param_names']['decorr_amp']]
                d_nu01 = params[comp['decorr_param_names']['decorr_nu01']]
                d_nu02 = params[comp['decorr_param_names']['decorr_nu02']]
                decorr_delta = d_amp**(1./np.log(d_nu01/d_nu02)**2)
                for f1 in range(self.nfreqs):
                    for f2 in range(f1, self.nfreqs):
                        sed_12 = decorrelated_bpass(self.bpss[f1],
                                                    self.bpss[f2],
                                                    sed, params,
                                                    decorr_delta)
                        comp_scaling[i_c, f1, f2] = sed_12 * units * units
            else:
                comp_scaling[i_c] = np.outer(single_sed[i_c], single_sed[i_c])

        for i_c1, c_name1 in enumerate(self.fg_model.component_names):
            fg_scaling[i_c1, i_c1] = comp_scaling[i_c1]
            for c_name2, epsname in self.fg_model.components[c_name1]['names_x_dict'].items():  # noqa
                i_c2 = self.fg_model.component_order[c_name2]
                eps = params[epsname]
                fg_scaling[i_c1, i_c2] = eps * np.outer(single_sed[i_c1],
                                                        single_sed[i_c2])
                fg_scaling[i_c2, i_c1] = eps * np.outer(single_sed[i_c2],
                                                        single_sed[i_c1])
        return fg_scaling, np.array(rot_matrices)

    def evaluate_power_spectra(self, params):
        # Evaluates the power spectra for each component and each ell
        #
        # Input:
        #   params: a dictionary of {NAME:VAL} pairs for the parameters. The power spectra will be evaluated at these ells
        #
        # Output:
        #   fg_pspecrtra: the foreground power spectra for each component, polarization and ell. Indicies represent, in order:
        #       [component][polarization1][polarization2][ell]
        fg_pspectra = np.zeros([self.fg_model.n_components, self.npol,
                                self.npol, self.n_ell])

        # Fill diagonal
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            for cl_comb, clfunc in comp['cl'].items():
                m1, m2 = cl_comb
                ip1 = self.pol_order[m1]
                ip2 = self.pol_order[m2]
                pspec_params = [params[comp['names_cl_dict'][cl_comb][k]]
                                for k in clfunc.params]
                p_spec = clfunc.eval(self.bpw_l, *pspec_params) * self.dl2cl
                fg_pspectra[i_c, ip1, ip2] = p_spec
                if m1 != m2:
                    fg_pspectra[i_c, ip2, ip1] = p_spec

        return fg_pspectra

    def model(self, params):
        # 
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   theoretical values of D_ell for all maps, for all ranges of ell sampled. 
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization), then the second map  
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #
        """
        Defines the total model and integrates over
        the bandpasses and windows.
        Parameters are 
        """
        # [npol,npol,nell]
        cmb_cell = (params['r_tensor'] * self.cmb_tens +
                    params['A_lens'] * self.cmb_lens +
                    self.cmb_scal) * self.dl2cl
        # [nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes=[2, 0, 1])
        
        # Kevin: this can be ignored for the standard case
        if self.config['cmb_model'].get('use_birefringence'):
            bi_angle = np.radians(params['birefringence'])
            c = np.cos(2*bi_angle)
            s = np.sin(2*bi_angle)
            bmat = np.array([[c, s],
                             [-s, c]])
            cmb_cell = rotate_cells_mat(bmat, bmat, cmb_cell)

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
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
                cls = (rotate_cells_mat(cmb_rot[f2], cmb_rot[f1], cmb_cell) *
                       cmb_scaling[f1] * cmb_scaling[f2])

                # Loop over component pairs
                for c1 in range(self.fg_model.n_components):
                    for c2 in range(self.fg_model.n_components):
                        mat1 = rot_m[c1, f1]
                        mat2 = rot_m[c2, f2]
                        if c1 == c2:
                            clrot = rotate_cells_mat(mat2, mat1, fg_cell[c1])
                        else:
                            # For cross component, enforcing EB term is zero.
                            cl_cross = np.zeros((self.n_ell,
                                                 self.npol, self.npol))
                            for i in range(self.npol):
                                cl_cross[:, i, i] = np.sqrt(
                                    fg_cell[c1, :, i, i] * fg_cell[c2, :, i, i]
                                )
                            clrot = rotate_cells_mat(mat2, mat1, cl_cross)
                        cls += clrot * fg_scaling[c1, c2, f1, f2]
                cls_array_fg[f1, f2] = cls

        # Add moment terms if needed
        if self.config['fg_model'].get('use_moments'):
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
            cls_11 = np.zeros([self.fg_model.n_components, self.n_ell,
                               self.npol, self.npol])
            # [ncomp, nell, npol, npol]
            cls_02 = np.zeros([self.fg_model.n_components, self.n_ell,
                               self.npol, self.npol])
            for i_c, c_name in enumerate(self.fg_model.component_names):
                comp = self.fg_model.components[c_name]
                gamma = params[comp['names_moments_dict']['gamma_beta']]
                amp = params[comp['names_moments_dict']['amp_beta']] * 1E-6
                cl_betas = self.bcls(lmax=lmax_mom, gamma=gamma, amp=amp)
                cl_cc = fg_cell[i_c, :]
                # cls_1x1 = 0
                cls_1x1 = self.evaluate_1x1(params, lmax=lmax_mom,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_11[i_c, :lmax_mom, :, :] = cls_1x1
                # cls_0x2 = 0
                cls_0x2 = self.evaluate_0x2(params, lmax=lmax_mom,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_02[i_c, :lmax_mom, :, :] = cls_0x2

            # Add components scaled in frequency
            for f1 in range(self.nfreqs):
                # Note that we only need to fill in half of the frequencies
                for f2 in range(f1, self.nfreqs):
                    cls = np.zeros([self.n_ell, self.npol, self.npol])
                    for c1 in range(self.fg_model.n_components):
                        cls += (fg_scaling_d1[f1, c1] * fg_scaling_d1[f2, c1] *
                                cls_11[c1])
                        cls += 0.5 * (fg_scaling_d2[f1, c1] *
                                      (fg_scaling[c1, c1, f2, f2])**0.5 +
                                      fg_scaling_d2[f2, c1] *
                                      (fg_scaling[c1, c1, f1, f1])**0.5) * cls_02[c1]  # noqa
                    cls_array_fg[f1, f2] += cls

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        # Polarization angle rotation
        for f1 in range(self.nfreqs):
            for f2 in range(self.nfreqs):
                cls_array_list[:, f1, :, f2, :] = rotate_cells(
                    self.bpss[f2], self.bpss[f1],
                    cls_array_list[:, f1, :, f2, :], params
                )

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def bcls(self, lmax, gamma, amp):
        ls = np.arange(lmax)
        bcls = np.zeros(len(ls))
        bcls[2:] = (ls[2:] / 80.)**gamma
        return bcls*amp

    def integrate_seds_der(self, params, order=1):
        """
        Define the first order derivative of the SED
        """
        fg_scaling_der = np.zeros([self.fg_model.n_components,
                                   self.nfreqs])

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]

            # Set SED function with scaling beta
            def sed_der(nu):
                nu0 = params[comp['names_sed_dict']['nu0']]
                x = np.log(nu / nu0)
                # This is only valid for spectral indices
                return x**order * comp['sed'].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b = self.bpss[tn].convolve_sed(sed_der, params)[0]
                fg_scaling_der[i_c, tn] = sed_b * units

        return fg_scaling_der.T

    def evaluate_1x1(self, params, lmax, cls_cc, cls_bb):
        """
        Evaluate the 1x1 moment for auto-spectra
        """

        ls = np.arange(lmax)
        v_left = (2*ls+1)[:, None, None] * cls_cc[:lmax, :, :]
        v_right = (2*ls+1) * cls_bb[:lmax]

        mat = self.big_w3j
        v_left = np.transpose(v_left, axes=[1, 0, 2])
        moment1x1 = np.dot(np.dot(mat, v_right), v_left) / (4*np.pi)
        return moment1x1

    def evaluate_0x2(self, params, lmax, cls_cc, cls_bb):
        """
        Evaluate the 0x2 moment for auto-spectra
        Assume power law for beta
        """
        ls = np.arange(lmax)
        prefac = np.sum((2 * ls + 1) * cls_bb) / (4*np.pi)
        return cls_cc[:lmax] * prefac

    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood.
        """
        model_cls = self.model(params)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    def prepare_h_and_l(self):
        """
        Prepare the HL likelihood.
        """
        from scipy.linalg import sqrtm  # noqa
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
        self.observed_cls = self.bbdata + self.bbnoise
        return

    def h_and_l_dx(self, params):
        """
        Hamimeche and Lewis likelihood.
        Taken from Cobaya written by H, L and Torrado
        See: https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/_cmblikes_prototype/_cmblikes_prototype.py  # noqa
        """
        model_cls = self.model(params)
        dx_vec = []
        for k in range(model_cls.shape[0]):
            C = model_cls[k] + self.bbnoise[k]
            X = self.h_and_l(C, self.observed_cls[k], self.Cfl_sqrt[k])
            if np.any(np.isinf(X)):
                return [np.inf]
            dx = self.matrix_to_vector(X).flatten()
            dx_vec = np.concatenate([dx_vec, dx])
        return dx_vec

    def h_and_l(self, C, Chat, Cfl_sqrt):
        """
        Evaluate the Hamimeche and Lewis likelihood.
        """
        try:
            diag, U = np.linalg.eigh(C)
        except:  # noqa
            return [np.inf]
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        try:
            diag, rot = np.linalg.eigh(rot)
        except:  # noqa
            return [np.inf]
        diag = (np.sign(diag - 1) *
                np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1)))
        Cfl_sqrt.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        return rot.dot(U.T)

    def lnprob(self, par):
        """
        Likelihood with priors.
        """
        prior = self.params.lnprior(self.get_jeffreys_numerical, par)
        if not np.isfinite(prior):
            return -np.inf

        return prior + self.lnlike(par)

    def lnlike(self, par):
        """
        Likelihood without priors.
        """
        params = self.params.build_params(par)
        if self.use_handl:
            dx = self.h_and_l_dx(params)
            if np.any(np.isinf(dx)):
                return -np.inf
        else:
            dx = self.chi_sq_dx(params)
        like = -0.5 * np.dot(dx, np.dot(self.invcov, dx))

        return like

    def emcee_sampler(self):
        """
        Sample the model with MCMC.
        """
        import emcee  # noqa
        from multiprocessing import Pool

        fname_temp = self.output_dir + '/emcee.npz.h5'
        backend = emcee.backends.HDFBackend(fname_temp)

        nwalkers = self.config['nwalkers']
        n_iters = self.config['n_iters']
        ndim = len(self.params.p0)
        found_file = os.path.isfile(fname_temp)

        try:
            nchain = len(backend.get_chain())
        except AttributeError:
            found_file = False

        if found_file and self.config['resume']:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(n_iters-nchain, 0)
        else:
            backend.reset(nwalkers, ndim)
            pos = [self.params.p0 + 1.e-3*np.random.randn(ndim)
                   for i in range(nwalkers)]
            nsteps_use = n_iters

        with Pool() as pool:  # noqa
            import time
            start = time.time()
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            self.lnprob,
                                            backend=backend)
            if nsteps_use > 0:
                sampler.run_mcmc(pos, nsteps_use, store=True, progress=True)
                end = time.time()

        return sampler, end-start

    def polychord_sampler(self):
        import pypolychord  # noqa
        from pypolychord.settings import PolyChordSettings  # noqa
        from pypolychord.priors import UniformPrior, GaussianPrior  # noqa

        ndim = len(self.params.p0)
        nder = 0

        # Log-likelihood compliant with PolyChord's input
        def likelihood(theta):
            return self.lnlike(theta), [0]

        def prior(hypercube):
            prior = []
            for h, pr in zip(hypercube, self.params.p_free_priors):
                if pr[1] == 'Gaussian':
                    prior.append(
                        GaussianPrior(float(pr[2][0]), float(pr[2][1]))(h)
                    )
                else:
                    prior.append(
                        UniformPrior(float(pr[2][0]), float(pr[2][2]))(h)
                    )
            return prior

        # Optional dumper function giving run-time read access to
        # the live points, dead points, weights and evidences
        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead point:", dead[-1])

        settings = PolyChordSettings(ndim, nder)
        settings.base_dir = self.output_dir + '/polychord'
        os.makedirs(settings.base_dir, exist_ok=True)
        settings.file_root = 'pch'
        settings.nlive = self.config['nlive']
        settings.num_repeats = self.config['nrepeat']
        settings.do_clustering = False  # Assume unimodal posterior
        settings.boost_posterior = 10   # Increase number of posterior samples
        settings.nprior = 200           # Draw nprior initial prior samples
        settings.maximise = True        # Maximize posterior at the end
        settings.read_resume = self.config['resume']   # Resume for earlier run
        settings.feedback = 0           # Verbosity {0,1,2,3}

        output = pypolychord.run_polychord(likelihood, ndim, nder, settings,
                                           prior, dumper)

        return output

    def minimizer(self):
        """
        Find maximum likelihood
        """
        from scipy.optimize import minimize  # noqa

        def chi2(par):
            c2 = -2*self.lnprob(par)
            return c2

        res = minimize(chi2, self.params.p0, method="Powell")
        return res.x

    def fisher(self):
        """
        Evaluate Fisher matrix
        """
        import numdifftools as nd  # noqa
        from scipy.optimize import minimize  # noqa

        def chi2(par):
            c2 = -2*self.lnprob(par)
            return c2

        res = minimize(chi2, self.params.p0, method="Powell")

        def lnprobd(p):
            ll = self.lnprob(p)
            if ll == -np.inf:
                ll = -1E100
            return ll

        fisher = - nd.Hessian(lnprobd)(res.x)
        return res.x, fisher

    def singlepoint(self):
        """
        Evaluate at a single point
        """
        chi2 = -2 * self.lnprob(self.params.p0)
        return chi2

    def timing(self, n_eval=300):
        """
        Evaluate n times and benchmark
        """
        import time
        start = time.time()
        for i in range(n_eval):
            self.lnprob(self.params.p0)
        end = time.time()

        return end-start, (end-start)/n_eval

    def predicted_spectra(self, at_min=True, save_npz=True):
        """
        Evaluates model at a the maximum likelihood and
        writes predicted spectra into a numpy array
        with shape (nbpws, nmaps, nmaps).
        """
        if at_min:
            sampler = self.minimizer()
            p = np.array(sampler)
        else:
            p = self.params.p0
        pars = self.params.build_params(p)
        model_cls = self.model(pars)
        tr_names = list(self.config['map_sets'].keys())

        if save_npz:
            np.savez(self.output_dir+'/cells_model.npz',
                     tracers=tr_names,
                     ls=self.ell_b,
                     dls=model_cls)
            return
        s = sacc.Sacc()
        for tn in tr_names:
            t = self.s.tracers[tn]
            s.add_tracer('NuMap', tn, quantity='cmb_polarization',
                         spin=2, nu=t.nu, bandpass=t.bandpass,
                         ell=t.ell, beam=t.beam, nu_unit='GHz',
                         map_unit='uK_CMB')
        for b1, b2, p1, p2, m1, m2, ind in self._freq_pol_iterator():
            cl = model_cls[:, m1, m2]
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cltyp = f'cl_{pol1}{pol2}'
            win = sacc.BandpowerWindow(self.bpw_l, self.windows[ind].T)
            s.add_ell_cl(cltyp, b1, b2, self.ell_b, cl, window=win)

        s.add_covariance(self.bbcovar)
        s.save_fits(self.output_dir+'/cells_model.fits',
                    overwrite=True)
        return

    def run(self):
        """
        Run the BB component separation stage.
        """
        # Make output directory
        self.output_dir = self.config["output_dir"].format(sim_id=self.sim_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Make duplicate of config file
        from shutil import copyfile
        copyfile(self.config_fname,
                 self.config["config_copy"].format(sim_id=self.sim_id))
        self.setup_compsep()

        # Get the chi2 and best-fit estimates
        from scipy.stats import chi2 as scipy_chi2
        sampler = self.minimizer()
        chi2 = -2*self.lnprob(sampler)
        ndof = len(self.bbcovar)
        kwargs = {
            "params": sampler,
            "names": self.params.p_free_names,
            "chi2": chi2,
            "ndof": len(self.bbcovar),
            "pte": scipy_chi2.sf(chi2, ndof, loc=0, scale=1)
        }
        np.savez(self.output_dir+'/chi2.npz', **kwargs)
        with open(self.output_dir+'/chi2.txt', 'w') as f:
            for key, value in kwargs.items():
                f.write('%s: %s\n' % (key, value))
        print("Saved best-fit parameters")

        # Get best-fit power spectra
        at_min = self.config.get('predict_at_minimum', True)
        save_npz = not self.config.get('predict_to_sacc', False)
        sampler = self.predicted_spectra(at_min=at_min, save_npz=save_npz)
        print("Predicted spectra saved")

        if self.config.get('sampler') == 'emcee':
            sampler, timing = self.emcee_sampler()
            np.savez(self.output_dir+'/emcee.npz',
                     chain=sampler.chain,
                     names=self.params.p_free_names,
                     time=timing)
            print("Finished sampling", timing)

        elif self.config.get('sampler') == 'polychord':
            os.makedirs(self.output_dir + '/polychord/clusters', exist_ok=True)
            sampler = self.polychord_sampler()
            print("Finished sampling")

        elif self.config.get('sampler') == 'fisher':
            p0, fisher = self.fisher()
            cov = np.linalg.inv(fisher)
            for i, (n, p) in enumerate(zip(self.params.p_free_names, p0)):
                print(n+" = %.3lE +- %.3lE" % (p, np.sqrt(cov[i, i])))
            np.savez(self.output_dir+'/fisher.npz',
                     params=p0, fisher=fisher,
                     names=self.params.p_free_names)

        elif self.config.get('sampler') == 'single_point':
            sampler = self.singlepoint()
            np.savez(self.output_dir+'/single_point.npz',
                     chi2=sampler, ndof=len(self.bbcovar),
                     names=self.params.p_free_names)
            print("Chi2:", sampler, len(self.bbcovar))

        elif self.config.get('sampler') == 'timing':
            sampler = self.timing()
            np.savez(self.output_dir+'/timing.npz',
                     timing=sampler[1],
                     names=self.params.p_free_names)
            print("Total time:", sampler[0])
            print("Time per eval:", sampler[1])
        else:
            raise ValueError("Unknown sampler")

        return

# Calculates the Jeffreys prior using numerical derivatives

    def get_jeffreys_numerical(self, par_names, par):
        # Calculates the jeffreys prior numerically, assuming a gaussian likelyhood
        #
        # input:
        #   par_names: the names of the parameters which will be given Jeffreys priors. All other parameters are assumed to have independent priors. 
        #   par: a list of the values of all free parameters. Jeffreys prior will be calculated at this point. 
        #
        # output: 
        #   a float indicating the value of the jeffrey's prior
        params = dict(zip(self.params.p_free_names,par)) | dict(self.params.p_fixed)
        return np.sqrt(np.linalg.det(self.get_fisher_numerical(par_names, params)))

    def get_fisher_numerical(self, par_names, params):
        # Calculates the Fisher matrix for the parameters listed in par_names, ignoring the other parameters
        #
        # Input:
        #   get_partials: function that calculates the partial derivatives of the model; defined in BBCompSep file
        #   par_names: a list of the names of parameters which we will calculate the fisher matrix of
        #   params: dictionary of {NAME:VAL} pairs for all of the parameters in the model. The fisher's matrix will be calculated at this point. 
        
        # [len(par_names), ell, nmap, nmap]
        partials = self.get_partials_numerical(par_names, params)
        partials = self.matrix_to_vector(partials)
        partials = partials.reshape([len(par_names),self.ncross * self.n_bpws])
        # [len(par_names), n_bpws * ncross]

        F = np.matmul(np.matmul(partials,self.invcov), np.transpose(partials))
        return F

    def get_partials_numerical(self, par ,params):
        # Returns a list of partial derivatives of the model
        #
        # Input: 
        #   model: the function to differentiate
        #   par: a list of the names of parameters that we wish to differentiate with respect to 
        #   params: a dictionary of {NAME:VAL} pairs for all of the model's parameters. The partial derivative will be taken about this point. 
        #   
        # Output:
        #   partials: a list of parial derivatives of the model. partials[i] is the partial derivative of the model with respect to par[i].
        partials = [self.get_partial_numerical(par_1,params) for par_1 in par]
        return np.array(partials)

    def get_partial_numerical(self, par_1, params):
        # Calculates the partial derivative of the model with respect to par_1, while params are kept constant. 
        #
        # Inputs:
        #   model: the function to differentiate
        #   par_1: the name of the parameter to differentiate with respect to
        #   params: a dictionary giving {NAME: VAL} pairs for all parameters of the model. The partial derivative will be evaluated at this point
        #
        # Output:
        #   derivative: an array, with the same dimentions as model, and indicies having the same meaning:
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        fixed_params = params.copy()
        par_1_val = fixed_params.pop(par_1)
        f = lambda x: self.model({par_1:x}|fixed_params)
        return self.derivative(f,par_1_val)

    def derivative(self, f,x):
        return (f(x+10**(-6))-f(x-10**(-6))) * 5*10**5

# Calculates the Jeffrey's prior using analytical derivatives
# TODO: analytic partial derivatives only work with the most basic model - they don't take into account bifringence, moments and rotations

    def get_jeffreys_analytical(self, par_names, par):
        # Calculates the jeffreys prior analytically, assuming a gaussian likelyhood
        #
        # input:
        #   par_names: the names of the parameters which will be given Jeffreys priors. All other parameters are assumed to have independent priors. 
        #   par: a list of the values of all free parameters. Jeffreys prior will be calculated at this point. 
        #
        # output: 
        #   a float indicating the value of the jeffrey's prior
        params = dict(zip(self.params.p_free_names,par)) | dict(self.params.p_fixed)
        return np.sqrt(np.linalg.det(self.get_fisher_analytical(par_names, params)))

    def get_fisher_analytical(self, par_names, params):
        # Calculates the Fisher matrix for the parameters listed in par_names, ignoring the other parameters
        #
        # Input:
        #   get_partials: function that calculates the partial derivatives of the model; defined in BBCompSep file
        #   par_names: a list of the names of parameters which we will calculate the fisher matrix of
        #   params: dictionary of {NAME:VAL} pairs for all of the parameters in the model. The fisher's matrix will be calculated at this point. 
        
        # [len(par_names), ell, nmap, nmap]
        partials = self.get_partials_analytical(par_names, params)
        partials = self.matrix_to_vector(partials)
        partials = partials.reshape([len(par_names),self.ncross * self.n_bpws])
        # [len(par_names), n_bpws * ncross]

        F = np.matmul(np.matmul(partials,self.invcov), np.transpose(partials))
        return F

    def get_partials_analytical(self, par ,params):
        # Returns a list of partial derivatives of the model
        #
        # Input: 
        #   model: the function to differentiate
        #   par: a list of the names of parameters that we wish to differentiate with respect to 
        #   params: a dictionary of {NAME:VAL} pairs for all of the model's parameters. The partial derivative will be taken about this point. 
        #   
        # Output:
        #   partials: a list of parial derivatives of the model. partials[i] is the partial derivative of the model with respect to par[i].
        partials = [self.get_partial_analytical(par_1,params) for par_1 in par]
        return np.array(partials)

    def get_partial_analytical(self, par_1, params):
        match par_1:
            case "r_tensor":
                return self.d_r_tensor(params)
            case "A_lens":
                return self.d_A_lens(params)
            case "amp_d_bb":
                return self.d_amp_d_bb(params)
            case "amp_s_bb":
                return self.d_amp_s_bb(params)
            case "epsilon_ds":
                return self.d_epsilon_ds(params)
            case "alpha_d_bb":
                return self.d_alpha_d_bb(params)
            case "alpha_s_bb":
                return self.d_alpha_s_bb(params)
            case "beta_s":
                return self.d_beta_s(params)
            case "beta_d":
                return self.d_beta_d(params)

    def d_r_tensor(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # [npol,npol,nell]
        d_cmb_cell = self.cmb_tens * self.dl2cl
        # [nell,npol,npol]
        d_cmb_cell = np.transpose(d_cmb_cell, axes=[2, 0, 1])
        
        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])

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
                cls = (rotate_cells_mat(cmb_rot[f2], cmb_rot[f1], d_cmb_cell) *
                       cmb_scaling[f1] * cmb_scaling[f2])
                cls_array_fg[f1, f2] = cls

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_A_lens(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        # TODO: bifringence, moments and rotations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # [npol,npol,nell]
        d_cmb_cell = self.cmb_lens * self.dl2cl
        # [nell,npol,npol]
        d_cmb_cell = np.transpose(d_cmb_cell, axes=[2, 0, 1])
        
        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])

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
                cls = (rotate_cells_mat(cmb_rot[f2], cmb_rot[f1], d_cmb_cell) *
                       cmb_scaling[f1] * cmb_scaling[f2])
                cls_array_fg[f1, f2] = cls

        # TODO: Moments
        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_amp_d_bb(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        # TODO: bifringence, moments and rotations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                mat1 = rot_m[0, f1]
                mat2 = rot_m[0, f2]
                clrot = rotate_cells_mat(mat2, mat1, fg_cell[0])/params["amp_d_bb"]
                cls = clrot * fg_scaling[0, 0, f1, f2]


                mat1 = rot_m[0, f1]
                mat2 = rot_m[1, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[0, :, 0, 0] * fg_cell[1, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(2*params["amp_d_bb"])
                cls += clrot * fg_scaling[0, 1, f1, f2]
                

                mat1 = rot_m[1, f1]
                mat2 = rot_m[0, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[1, :, 0, 0] * fg_cell[0, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(2*params["amp_d_bb"])
                cls += clrot * fg_scaling[1, 0, f1, f2]

                cls_array_fg[f1, f2] = cls

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_amp_s_bb(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        # TODO: bifringence, moments and rotations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                mat1 = rot_m[1, f1]
                mat2 = rot_m[1, f2]
                clrot = rotate_cells_mat(mat2, mat1, fg_cell[1])/params["amp_s_bb"]
                cls = clrot * fg_scaling[1, 1, f1, f2]


                mat1 = rot_m[0, f1]
                mat2 = rot_m[1, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[0, :, 0, 0] * fg_cell[1, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(2*params["amp_s_bb"])
                cls += clrot * fg_scaling[0, 1, f1, f2]
                

                mat1 = rot_m[1, f1]
                mat2 = rot_m[0, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[1, :, 0, 0] * fg_cell[0, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(2*params["amp_s_bb"])
                cls += clrot * fg_scaling[1, 0, f1, f2]

                cls_array_fg[f1, f2] = cls

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_epsilon_ds(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        # TODO: bifringence, moments and rotations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):

                mat1 = rot_m[0, f1]
                mat2 = rot_m[1, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[0, :, 0, 0] * fg_cell[1, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(params["epsilon_ds"])
                cls = clrot * fg_scaling[0, 1, f1, f2]
                

                mat1 = rot_m[1, f1]
                mat2 = rot_m[0, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[1, :, 0, 0] * fg_cell[0, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/(params["epsilon_ds"])
                cls += clrot * fg_scaling[1, 0, f1, f2]

                cls_array_fg[f1, f2] = cls

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_alpha_d_bb(self, params):
        # Calculates the partial derivative of model with respect to alpha_d_bb
        # TODO: bifringence, moments and rotations
        # TODO: code currently relies on ell_0 = 80
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                mat1 = rot_m[0, f1]
                mat2 = rot_m[0, f2]
                clrot = rotate_cells_mat(mat2, mat1, fg_cell[0])
                cls = clrot * fg_scaling[0, 0, f1, f2]


                mat1 = rot_m[0, f1]
                mat2 = rot_m[1, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[0, :, 0, 0] * fg_cell[1, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/2
                cls += clrot * fg_scaling[0, 1, f1, f2]
                

                mat1 = rot_m[1, f1]
                mat2 = rot_m[0, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[1, :, 0, 0] * fg_cell[0, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/2
                cls += clrot * fg_scaling[1, 0, f1, f2]

                cls_array_fg[f1, f2] = cls

        # Complete the calculation (need to multiply by log(ell/ell_0))
        ells = self.bpw_l.reshape(1,1,-1,1,1)
        cls_array_fg = cls_array_fg * np.log(ells/80)

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_alpha_s_bb(self, params):
        # Calculates the partial derivative of model with respect to r_tensor
        # TODO: bifringence, moments and rotations
        # TODO: code currently relies on ell_0 = 80
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   partial derivative of the model (with the same output format as model())
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization, 
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # Kevin: This is evaluating the foreground SEDs at the SO channels'
        # frequencies
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                mat1 = rot_m[1, f1]
                mat2 = rot_m[1, f2]
                clrot = rotate_cells_mat(mat2, mat1, fg_cell[1])
                cls = clrot * fg_scaling[1, 1, f1, f2]


                mat1 = rot_m[0, f1]
                mat2 = rot_m[1, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[0, :, 0, 0] * fg_cell[1, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/2
                cls += clrot * fg_scaling[0, 1, f1, f2]
                

                mat1 = rot_m[1, f1]
                mat2 = rot_m[0, f2]
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                cl_cross[:, 0, 0] = np.sqrt(
                    fg_cell[1, :, 0, 0] * fg_cell[0, :, 0, 0]
                )
                clrot = rotate_cells_mat(mat2, mat1, cl_cross)/2
                cls += clrot * fg_scaling[1, 0, f1, f2]

                cls_array_fg[f1, f2] = cls

        # Complete the calculation (need to multiply by log(ell/ell_0))
        ells = self.bpw_l.reshape(1,1,-1,1,1)
        cls_array_fg = cls_array_fg * np.log(ells/80)

        # === Theory model computation ends here (except polarization angle
        #     rotation) ===
        # * Coadded CMB + dust + synchrotron D_ells: cls_array_fg
        #   (sorry for misleading name!)
        #   shape: [nfreqs, nfreqs, n_ell, npol, npol]

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def integrate_sed_derivatives(self, params):
        # Calculates the derivatives of the SED with respect to the betas (assuming an exponential dependence on beta, (nu/nu_0)^beta )
        #
        # Input:
        #   params: a dictionary of {NAME: VAL} pairs for the parameters. sed is evaluated at these values
        #
        # Output:
        #   fgscaling: a 4d array containing SED scalings for each component and frequency. Indecies represent (in order):
        #       [component1][component2][freq_band1][freq_band2]
        #   rot_matricies: a 1d array of rotation matricies for each frequency

        single_sed = np.zeros([self.fg_model.n_components,
                               self.nfreqs])
        single_d_sed = np.zeros([self.fg_model.n_components,
                               self.nfreqs])

        for i_c, c_name in enumerate(self.fg_model.component_names):

            comp = self.fg_model.components[c_name]
            nu0 = comp["nu0"]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]

            sed = lambda nu: comp['sed'].eval(nu, *sed_params)
            d_sed = lambda nu: comp['sed'].eval(nu, *sed_params) * np.log(nu/nu0)

            for tn in range(self.nfreqs):
                sed_b, _ = self.bpss[tn].convolve_sed(sed, params)
                single_sed[i_c, tn] = sed_b * units
                d_sed_b, _ = self.bpss[tn].convolve_sed(d_sed, params)
                single_d_sed[i_c, tn] = d_sed_b * units

        return single_sed, single_d_sed

    def d_beta_d(self, params):
        # TODO: brefringence, moments, roations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   theoretical values of D_ell for all maps, for all ranges of ell sampled. 
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization), then the second map  
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #

        # [ncomp,nfreq]
        sed, d_sed = self.integrate_sed_derivatives(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):

                # Derivative of D_l^{d,BB}
                clrot = fg_cell[0]
                cls = clrot * (sed[0][f1] * d_sed[0][f2] + sed[0][f2] * d_sed[0][f1])

                # Derivative of D_l^{sxd, BB}
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                for i in range(self.npol):
                
                    cl_cross[:, i, i] = np.sqrt(
                        fg_cell[1, :, i, i] * fg_cell[0, :, i, i]
                    )
                clrot = cl_cross
                cls += clrot * (sed[1][f1]*d_sed[0][f2] + sed[1][f2]*d_sed[0][f1]) * params["epsilon_ds"]

                cls_array_fg[f1, f2] = cls

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def d_beta_s(self, params):
        # TODO: brefringence, moments, roations
        #
        # Input:
        #   params: DICT of {NAME, VAL} pairs for all of the parameters. The model is evaluated at these values. 
        #
        # Output: 
        #   theoretical values of D_ell for all maps, for all ranges of ell sampled. 
        #   Indicies are (in order) bpws (i.e. the range of ells), the first map (columns are arranged first by band then by polarization), then the second map  
        #   e.g. if we had two bands b1 and b2 and two polarizeraitons E and B, then the columns would correspond to b1E b1B b2E b2B)
        #
        """
        Defines the total model and integrates over
        the bandpasses and windows.
        Parameters are 
        """

        # [ncomp,nfreq]
        sed, d_sed = self.integrate_sed_derivatives(params)

        # Kevin: This is the power law model for foreground power spectra
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):

                # Derivative of D_l^{d,BB}
                clrot = fg_cell[1]
                cls = clrot * (sed[1][f1] * d_sed[1][f2] + sed[1][f2] * d_sed[1][f1])

                # Derivative of D_l^{sxd, BB}
                cl_cross = np.zeros((self.n_ell,
                                        self.npol, self.npol))
                for i in range(self.npol):
                    cl_cross[:, i, i] = np.sqrt(
                        fg_cell[1, :, i, i] * fg_cell[0, :, i, i]
                    )
                clrot = cl_cross
                cls += clrot * (sed[0][f1]*d_sed[1][f2] + sed[0][f2]*d_sed[1][f1]) * params["epsilon_ds"]

                cls_array_fg[f1, f2] = cls

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])


def main(args):
    """
    Execute the BBCompSep stage with arguments args:
        * config: string.
          Path to configuration file with input parameters.
    """
    config = _yaml_loader(args.config)

    # Creating the simulation indices range to loop over
    sim_ids = config["global"]["sim_ids"]
    if isinstance(sim_ids, list):
        sim_ids = np.array(sim_ids, dtype=int)
    elif isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])
    else:
        sim_ids = [None]

    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Initialize tasks for MPI sharing
    mpi_shared_list = sim_ids

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id in local_mpi_list:
        start = time.time()
        compsep = BBCompSep(args)
        setattr(compsep, "sim_id", sim_id)
        compsep.run()
    mpi.print_rnk0(f"Processed {len(sim_ids)} simulations "
                   f"in {time.time() - start:.1f} seconds.", rank)
    comm.Barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SO SAT BB likelihood")
    parser.add_argument(
        "--config", type=str,
        help="Path to yaml file with pipeline configuration"
    )

    args = parser.parse_args()
    main(args)

class NotConvergedError(Exception):
    pass
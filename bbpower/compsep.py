import numpy as np
import os
from scipy.linalg import sqrtm

from bbpipe import PipelineStage
from .types import NpzFile, FitsFile, YamlFile
from .fg_model import FGModel
from .param_manager import ParameterManager
from .bandpasses import Bandpass, rotate_cells, rotate_cells_mat
from fgbuster.component_model import CMB 
import sacc

class BBCompSep(PipelineStage):
    """
    Component separation stage
    This stage does harmonic domain foreground cleaning (e.g. BICEP).
    The foreground model parameters are defined in the config.yml file. 
    """
    name = "BBCompSep"
    inputs = [('cells_coadded', FitsFile),
              ('cells_noise', FitsFile),
              ('cells_fiducial', FitsFile),
              ('cells_coadded_cov', FitsFile)] # Read covariance from separate file
    outputs = [('param_chains', NpzFile),
               ('config_copy', YamlFile)]
    config_options={'likelihood_type':'h&l', 'n_iters':32, 'nwalkers':16, 'r_init':1.e-3,
                    'sampler':'emcee', 'bands': 'all'}

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        if self.config['fg_model'].get('use_moments'):
            print('Running moments')
            self.precompute_w3j()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.params = ParameterManager(self.config)
        if self.config.get("diff"):
            print('Running hybrid')            
            self.hybridparams = np.load(self.config['resid_seds'])
            hyb_params = self.hybridparams['params_cent']
            hyb_sigma = self.hybridparams['params_sigma']
            # Beta_s: [centre, sigma]
            self.params.p_free_priors[6][2] = [hyb_params[0], hyb_sigma[0]]
            # Beta_d: [centre, sigma]
            self.params.p_free_priors[2][2] = [hyb_params[1], hyb_sigma[1]]
            if hyb_sigma[0]==np.nan:
                hyb_sigma[0]=0.1
            if hyb_sigma[1]==np.nan:
                hyb_sigma[1]=0.1
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
        return mat[..., self.index_ut_modes[0], self.index_ut_modes[1]]

    def vector_to_matrix(self, vec):
        if vec.ndim == 1:
            mat = np.zeros([self.nmaps, self.nmaps])
            mat[self.index_ut] = vec
            mat = mat + mat.T - np.diag(mat.diagonal())
        elif vec.ndim==2:
            mat = np.zeros([len(vec), self.nmaps, self.nmaps])
            mat[..., self.index_ut[0], self.index_ut[1]] = vec[...,:]
            for i,m in enumerate(mat):
                mat[i] = m + m.T - np.diag(m.diagonal())
        else:
            raise ValueError("Input vector can only be 1- or 2-D")
        return mat

    def _freq_pol_iterator(self):
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

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra, 
        bandpasses, and window functions. 
        """
        # Decide if you're using H&L
        self.use_handl = self.config['likelihood_type'] == 'h&l'

        # Read data
        self.s = sacc.Sacc.load_fits(self.get_input('cells_coadded'))
        self.s_cov = sacc.Sacc.load_fits(self.get_input('cells_coadded_cov'))
        if self.use_handl:
            s_fid = sacc.Sacc.load_fits(self.get_input('cells_fiducial'))
            s_noi = sacc.Sacc.load_fits(self.get_input('cells_noise'))

        # Keep only desired correlations
        self.pols = self.config['pol_channels']
        corr_all = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        corr_keep = []
        for m1 in self.pols:
            for m2 in self.pols:
                clname = 'cl_' + m1.lower() + m2.lower()
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

        if self.config['bands'] == 'all':
            tr_names = sorted(list(self.s.tracers.keys()))
        else:
            tr_names = self.config['bands']
        self.nfreqs = len(tr_names)
        self.npol = len(self.pols)
        self.nmaps = self.nfreqs * self.npol
        self.index_ut = np.triu_indices(self.nmaps)
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.pol_order=dict(zip(self.pols,range(self.npol)))

        # Collect bandpasses
        self.bpss = []
        for i_t, tn in enumerate(tr_names):
            t = self.s.tracers[tn]
            nu = t.nu
            dnu = np.zeros_like(nu);
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
        if self.config['bands'] == 'all':
            if not (self.s_cov.covariance.covmat.shape[-1] == len(self.s.mean) == self.n_bpws * self.ncross): 
                raise ValueError("C_ell vector's size is wrong")
        v2d = np.zeros([self.n_bpws, self.ncross])
        if self.use_handl:
            v2d_noi = np.zeros([self.n_bpws, self.ncross])
            v2d_fid = np.zeros([self.n_bpws, self.ncross])
        cv2d = np.zeros([self.n_bpws, self.ncross, self.n_bpws, self.ncross])

        self.vector_indices = self.vector_to_matrix(np.arange(self.ncross, dtype=int)).astype(int)
        self.indx = []

        # Parse into the right ordering
        for b1, b2, p1, p2, m1, m2, ind_vec in self._freq_pol_iterator():
            t1 = tr_names[b1]
            t2 = tr_names[b2]
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cl_typ = f'cl_{pol1}{pol2}'
            ind_a = self.s.indices(cl_typ, (t1, t2))
            if len(ind_a) != self.n_bpws:
                raise ValueError("All power spectra need to be sampled at the same ells")
            w = self.s.get_bandpower_windows(ind_a)
            self.windows[ind_vec, :, :] = w.weight[mask_w, :].T
            v2d[:, ind_vec] = np.array(self.s.mean[ind_a])
            if self.use_handl:
                _, v2d_noi[:, ind_vec] = s_noi.get_ell_cl(cl_typ, t1, t2)
                _, v2d_fid[:, ind_vec] = s_fid.get_ell_cl(cl_typ, t1, t2)
            for b1b, b2b, p1b, p2b, m1b, m2b, ind_vecb in self._freq_pol_iterator():
                t1b = tr_names[b1b]
                t2b = tr_names[b2b]
                pol1b = self.pols[p1b].lower()
                pol2b = self.pols[p2b].lower()
                cl_typb = f'cl_{pol1b}{pol2b}'
                ind_b = self.s.indices(cl_typb, (t1b, t2b))
                cv2d[:, ind_vec, :, ind_vecb] = self.s_cov.covariance.covmat[ind_a][:, ind_b] 

        #Store data
        self.bbdata = self.vector_to_matrix(v2d) #(nbpw, nf, nf)
        if self.config.get("diff"):
            from scipy.linalg import eig, inv, pinv
            self.hybridparams = np.load(self.config['resid_seds'])
            Qmat = self.hybridparams['Q']
            eval, evec = eig(Qmat)
            mask = eval > 0.1 #take away rows equivalent to eigenvalue=0
            self.R = np.linalg.inv(evec)[mask, :]
            self.bbdata = np.einsum("aj,bk,ijk->iab", self.R, self.R, self.bbdata) 
            cov_all = np.zeros([self.n_bpws, self.nmaps, self.nmaps, self.n_bpws, self.nmaps, self.nmaps])
            for ix in range(self.ncross):
                i, j = np.array(self.index_ut)[:,ix] 
                for jx in range(self.ncross): 
                    m, n = np.array(self.index_ut)[:,jx] 
                    cov_all[:, i, j, :, m, n] = cv2d[:, ix, :, jx]
                    cov_all[:, j, i, :, m, n] = cv2d[:, ix, :, jx]
                    cov_all[:, i, j, :, n, m] = cv2d[:, ix, :, jx]
                    cov_all[:, j, i, :, n, m] = cv2d[:, ix, :, jx]
            ccov = np.einsum("aj,bk,gm,dn,ijklmn->iablgd", self.R, self.R, self.R, self.R, cov_all)
            self.nmodes = 4
            self.nmaps_modes = self.nmodes * self.npol
            self.index_ut_modes = np.triu_indices(self.nmodes)
            self.ncross_modes = (self.nmaps_modes * (self.nmaps_modes + 1)) // 2
            cv2d = np.zeros([self.n_bpws, self.ncross_modes, self.n_bpws, self.ncross_modes])
            for ix in range(self.ncross_modes):
                i, j = np.array(self.index_ut_modes)[:,ix]
                for jx in range(self.ncross_modes): 
                    m, n = np.array(self.index_ut_modes)[:,jx]
                    #import traceback
                    #traceback.print_stack()
                    cv2d[:, ix, :, jx] = ccov[:, i, j, :, m, n]
        else:
            self.nmodes = self.nfreqs
            self.nmaps_modes = self.nmaps
            self.index_ut_modes = self.index_ut
            self.ncross_modes = self.ncross
        if self.use_handl:
            self.bbnoise = self.vector_to_matrix(v2d_noi)
            self.bbfiducial = self.vector_to_matrix(v2d_fid)
        self.bbcovar = cv2d.reshape([self.n_bpws * self.ncross_modes,
                                     self.n_bpws * self.ncross_modes])
        self.invcov = np.linalg.inv(self.bbcovar)
        
        return
    
    def load_cmb(self):
        """
        Loads the CMB BB spectrum as defined in the config file. 
        """
        cmb_lensingfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][0])
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
            self.cmb_tens[ind, ind] = cmb_bbfile[:, 3][mask] - cmb_lensingfile[:, 3][mask]
            self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        if 'E' in self.config['pol_channels']:
            ind = self.pol_order['E']
            self.cmb_tens[ind, ind] = cmb_bbfile[:, 2][mask] - cmb_lensingfile[:, 2][mask]
            self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]
        return

    def integrate_seds(self, params): #NB different from NERSC
        fg_scaling = np.zeros([self.fg_model.n_components, self.nfreqs])
        rot_matrices = []

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]] 
                          for k in comp['sed'].params]
            rot_matrices.append([])
            
            if self.config.get("diff"):
                def sed(nu):
                    return comp['sed'].diff(nu, *sed_params) 
            else:
                def sed(nu):
                    return comp['sed'].eval(nu, *sed_params)
            
            for tn in range(self.nfreqs):
                sed_b, rot = self.bpss[tn].convolve_sed(sed, params)
                fg_scaling[i_c, tn] = sed_b * units
                rot_matrices[i_c].append(rot)

        return fg_scaling.T, rot_matrices

    def evaluate_power_spectra(self, params): #NB different from NERSC
        fg_pspectra = np.zeros([self.fg_model.n_components,
                                self.fg_model.n_components,
                                self.npol, self.npol, self.n_ell]) #NB different from NERSC
        
        # Fill diagonal
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            for cl_comb,clfunc in comp['cl'].items():
                m1, m2 = cl_comb
                ip1 = self.pol_order[m1]
                ip2 = self.pol_order[m2]
                pspec_params = [params[comp['names_cl_dict'][cl_comb][k]]
                                for k in clfunc.params]
                fg_pspectra[i_c, i_c, ip1, ip2, :] = clfunc.eval(self.bpw_l, *pspec_params) * self.dl2cl

        # Off diagonals #NB different from NERSC
        for i_c1, c_name1 in enumerate(self.fg_model.component_names):
            for c_name2, epsname in self.fg_model.components[c_name1]['names_x_dict'].items():
                i_c2 = self.fg_model.component_order[c_name2]
                cl_x=np.sqrt(np.fabs(fg_pspectra[i_c1, i_c1]*
                                     fg_pspectra[i_c2, i_c2])) * params[epsname]
                fg_pspectra[i_c1, i_c2] = cl_x
                fg_pspectra[i_c2, i_c1] = cl_x

        return fg_pspectra
    
    def model(self, params): #NB different from NERSC (dimensions)
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        # [npol,npol,nell]
        cmb_cell = (params['r_tensor'] * self.cmb_tens + \
                    params['A_lens'] * self.cmb_lens + \
                    self.cmb_scal) * self.dl2cl
        # [nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes = [2,0,1])
        if self.config['cmb_model'].get('use_birefringence'):
            bi_angle = np.radians(params['birefringence'])
            c = np.cos(2*bi_angle)
            s = np.sin(2*bi_angle)
            bmat = np.array([[c, s],
                             [-s, c]])
            cmb_cell = rotate_cells_mat(bmat, bmat, cmb_cell)        
        
        # [nfreq, ncomp], [ncomp,nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params) #NB different from NERSC (dimensions)
        # [ncomp,ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params) #NB different from NERSC (dimensions)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,ncomp,nell,npol,npol] 
        fg_cell = np.transpose(fg_cell, axes = [0,1,4,2,3])  #NB different from NERSC

        #NB different from NERSC no CMB scaling

        # Loop over frequencies
        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1,self.nfreqs):
                cls=cmb_cell.copy() #NB different from NERSC

                # Loop over component pairs
                for c1 in range(self.fg_model.n_components):
                    mat1=rot_m[c1][f1] #NB different from NERSC
                    a1=fg_scaling[f1,c1]
                    for c2 in range(self.fg_model.n_components):
                        mat2=rot_m[c2][f2]
                        a2=fg_scaling[f2,c2]
                        clrot=rotate_cells_mat(mat2,mat1,fg_cell[c1,c2])
                        cls += clrot*a1*a2
                cls_array_fg[f1,f2]=cls

        # Add moment terms if needed
        # Start by assuming that there is no cross-correlation between components and betas
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

            # component_1: dust, component_2: sync
            for i_c, c_name in enumerate(self.fg_model.component_names):
                comp = self.fg_model.components[c_name]
                gamma = params[comp['names_moments_dict']['gamma_beta']]
                amp = params[comp['names_moments_dict']['amp_beta']] * 1E-6
                cl_betas = self.bcls(lmax=lmax_mom, gamma=gamma, amp=amp)
                #cl_cc = fg_cell[i_c, :] #NERSC version
                cl_cc = fg_cell[i_c, i_c, :] #NB different from NERSC
                cls_1x1 = self.evaluate_1x1(params, lmax=lmax_mom,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_11[i_c, :lmax_mom, :, :] = cls_1x1

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
                        #NERSC version
                        #cls += 0.5 * (fg_scaling_d2[f1, c1] * 
                        #              fg_scaling[c1, c1, f2, f2] +
                        #              fg_scaling_d2[f2, c1] *
                        #              fg_scaling[c1, c1, f1, f1]) * cls_02[c1]
                        cls += 0.5 * (fg_scaling_d2[f1, c1] *
                                      fg_scaling[f2, c1] +
                                      fg_scaling_d2[f2, c1] *
                                      fg_scaling[f1, c1]) * cls_02[c1]
                    cls_array_fg[f1, f2] += cls

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs, self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1,self.nfreqs):
                    p0 = p1 if f1==f2 else 0
                    for p2 in range(p0,self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1,f2,:,p1,p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1!=m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        # Polarization angle rotation
        for f1 in range(self.nfreqs):
            for f2 in range(self.nfreqs):
                cls_array_list[:,f1,:,f2,:] = rotate_cells(self.bpss[f2],
                                                           self.bpss[f1],
                                                           cls_array_list[:,f1,:,f2,:],
                                                           params)
        cls_array_list = cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])
        if self.config.get("diff"):
            cls_array_list = np.einsum("aj,bk,ijk->iab", self.R, self.R, cls_array_list)
        return cls_array_list

    def bcls(self, lmax, gamma, amp):
        """
        Model beta spectra as power laws.
        """
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

    def evaluate_0x2(self, params, lmax, cls_cc, cls_bb,
                     gamma=None):
        """
        Evaluate the 0x2 moment for auto-spectra
        Assume power law for beta
        """
        ls = np.arange(lmax)
        if gamma is not None:
            prefac = amp * (2 * sp.zeta(-gamma-1) + sp.zeta(-gamma) - 3) / (4 * np.pi * 80**gamma)
        else:
            prefac = np.sum( (2 * ls + 1) * cls_bb) / (4*np.pi)
        return cls_cc[:lmax] * prefac

    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood. 
        """
        model_cls = self.model(params)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    def prepare_h_and_l(self):
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
        self.observed_cls = self.bbdata + self.bbnoise
        return 

    def h_and_l_dx(self, params):
        """                                                                                                 Hamimeche and Lewis likelihood.                                                                     Taken from Cobaya written by H, L and Torrado                                                       See: https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/_cmblikes_prototype/_cmblikes_prototype.py                                                        
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
        try:
            diag, U = np.linalg.eigh(C)
        except:
            return [np.inf]
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        try:
            diag, rot = np.linalg.eigh(rot)
        except:
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
        prior = self.params.lnprior(par)
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
        import emcee
        
        fname_temp = self.get_output('param_chains')+'.h5'
        backend = emcee.backends.HDFBackend(fname_temp)

        nwalkers = self.config['nwalkers']
        n_iters = self.config['n_iters']
        ndim = len(self.params.p0)
        found_file = os.path.isfile(fname_temp)

        if not found_file:
            backend.reset(nwalkers,ndim)
            p0 = self.minimizer()
            pos = [p0 + 1.e-3*np.random.randn(ndim) for i in range(nwalkers)]
            nsteps_use = n_iters
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(n_iters-len(backend.get_chain()), 0)

        import time
        start = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, backend=backend)
        if nsteps_use > 0:
            sampler.run_mcmc(pos, nsteps_use, store=True, progress=True);
        end = time.time()
        return sampler, end-start

    #NB different from NERSC: Add Polychord
    
    def minimizer(self):
        """
        Find maximum likelihood
        """
        from scipy.optimize import minimize
        def chi2(par):
            c2=-2*self.lnprob(par)
            return c2
        res=minimize(chi2, self.params.p0, method="Powell")

        if len(self.params.p0)<2:
            x = np.array([res.x])
        else:
            x = res.x
        return x        

    def fisher(self):
        """
        Evaluate Fisher matrix
        """
        import numdifftools as nd
        
        def lnprobd(p):
            l = self.lnprob(p)
            if l == -np.inf:
                l = -1E100
            return l
        fisher = - nd.Hessian(lnprobd)(self.params.p0)
        #NB different from NERSC: self.params.p0 given by chi^2 minim.
        return fisher

    def singlepoint(self):
        """
        Evaluate at a single point
        """
        chi2 = -2*self.lnprob(self.params.p0)
        return chi2

    def timing(self, n_eval=300):
        """
        Evaluate n times and benchmark
        """
        import time
        start = time.time()
        for i in range(n_eval):
            lik = self.lnprob(self.params.p0)
        end = time.time()
        return end-start, (end-start)/n_eval

    
    def run(self):
        from shutil import copyfile
        self.setup_compsep()
        if self.config.get('sampler')=='emcee':
            sampler, timing = self.emcee_sampler()
            np.savez(self.get_output('param_chains'),
                     names=self.params.p_free_names, time=timing)
            print("Finished sampling: ", timing)
        elif self.config.get('sampler')=='fisher':
            fisher = self.fisher()
            cov = np.linalg.inv(fisher)
            for i,(n,p) in enumerate(zip(self.params.p_free_names,
                                         self.params.p0)):
                print(n+" = %.3lE +- %.3lE" % (p, np.sqrt(cov[i, i])))
            np.savez(self.get_output('param_chains'),
                     fisher=fisher,
                     names=self.params.p_free_names)
        elif self.config.get('sampler')=='maximum_likelihood':
            sampler = self.minimizer()
            chi2 = -2*self.lnprob(sampler)
            np.savez(self.get_output('param_chains'),
                     params=sampler,
                     names=self.params.p_free_names,
                     chi2=chi2, ndof=len(self.bbcovar))
            print("Best fit:")
            for n,p in zip(self.params.p_free_names,sampler):
                print(n+" = %.3lE" % p)
            print("Chi2: %.3lE" % chi2)
        elif self.config.get('sampler')=='single_point':
            sampler = self.singlepoint()
            np.savez(self.get_output('param_chains'),
                     chi2=sampler, ndof=len(self.bbcovar),
                     names=self.params.p_free_names)
            print("Chi^2:",sampler,len(self.bbcovar))
        elif self.config.get('sampler')=='timing':
            sampler = self.timing()
            np.savez(self.get_output('param_chains'),
                     timing=sampler[1],
                     names=self.params.p_free_names)
            print("Total time:",sampler[0])
            print("Time per eval:",sampler[1])
        else:
            raise ValueError("Unknown sampler")

        return

if __name__ == '__main__':
    cls = PipelineStage.main()

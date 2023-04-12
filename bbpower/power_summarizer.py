from bbpipe import PipelineStage
from .types import TextFile, FitsFile
import sacc
import numpy as np


class BBPowerSummarizer(PipelineStage):
    name = "BBPowerSummarizer"
    inputs = [('splits_list', TextFile), ('bandpasses_list', TextFile),
              ('cells_all_splits', FitsFile), ('cells_all_sims', TextFile)]
    outputs = [('cells_coadded_total', FitsFile), ('cells_coadded', FitsFile),
               ('cells_noise', FitsFile), ('cells_null', FitsFile)]
    config_options = {'nulls_covar_type': 'diagonal',
                      'nulls_covar_diag_order': 0,
                      'data_covar_type': 'block_diagonal',
                      'data_covar_diag_order': 3}

    def get_covariance_from_samples(self, v, s, covar_type='dense',
                                    off_diagonal_cut=0):
        """
        Computes a covariance matrix from a set of samples in the form
        [nsamples, ndata]
        """
        if covar_type == 'diagonal':
            cov = np.diag(np.std(v, axis=0)**2)
        else:
            nsim, nd = v.shape
            vmean = np.mean(v, axis=0)
            cov = np.einsum('ij,ik', v, v)
            cov = cov/nsim - vmean[None, :]*vmean[:, None]
            if covar_type == 'block_diagonal':
                nblocks = nd // self.n_bpws
                cuts = np.ones([self.n_bpws, self.n_bpws])
                if nblocks * self.n_bpws != nd:
                    raise ValueError("Vector can't be divided into blocks")
                for i in range(off_diagonal_cut+1, self.n_bpws):
                    cuts -= np.diag(np.ones(self.n_bpws-i), k=i)
                    cuts -= np.diag(np.ones(self.n_bpws-i), k=-i)
                cov = cov.reshape([nblocks, self.n_bpws, nblocks, self.n_bpws])
                cov = (cov * cuts[None, :, None, :]).reshape([nd, nd])
        s.add_covariance(cov)

    def init_params(self):
        """
        Read some input files to determine the size of the power spectra
        """
        # Calculate number of splits and number of frequency channels
        self.nsplits = len(open(self.get_input('splits_list'),
                                'r').readlines())
        self.nbands = len(open(self.get_input('bandpasses_list'),
                               'r').readlines())
        self.delensing = self.config['delensing']
        self.n_channels_del = self.nbands+1
        if self.delensing:
            self.n_channels = self.n_channels_del
        else:
            self.n_channels = self.nbands
        # Compute all possible null combinations
        # Currently we compute these as (m_i-m_j) x (m_k-m_l)
        # where m_x is the set of maps for split x,
        # (i,j,k,l) are all different numbers.
        # Note that the actual number of possible nulls is actually infinite.
        # The most general form would be (sum_i a_i * m_i) x (sum_i b_i * m_i)
        # where both weight vectors a_i, b_i are orthogonal and have zero sum.
        # This way we're restricting ourselves to cases of the form:
        #    a = (1,-1,0,0), b=(0,0,1,-1)

        # First, figure out all possible pairings
        first_pairs = []
        self.pairings = []
        for i in range(self.nsplits):
            # Loop over js that aren't i
            listj = list(filter(lambda x: x not in [i],
                                range(self.nsplits)))
            for j in listj:
                if j < i:
                    continue
                first_pairs.append((i, j))
                # ks that aren't j or i
                listk = list(filter(lambda x: x not in [i, j],
                                    range(self.nsplits)))
                for k in listk:
                    # l != i,j,k
                    listl = list(filter(lambda x: x not in [i, j, k],
                                        range(self.nsplits)))
                    for l in listl:
                        if l < k:
                            continue
                        if (k, l) in first_pairs:
                            continue
                        self.pairings.append((i, j, k, l))
        self.n_nulls = len(self.pairings)

        # First, initialize n_bpws to zero
        self.n_bpws = 0
        # Read splits power spectra
        self.s_splits = sacc.Sacc.load_fits(self.get_input('cells_all_splits'))
        # Read sorting and number of bandpowers
        self.check_sacc_consistency(self.s_splits)
        # Read file names for the power spectra of all simulations
        with open(self.get_input('cells_all_sims')) as f:
            content = f.readlines()
        self.fname_sims = [x.strip() for x in content]
        self.nsims = len(self.fname_sims)
        # Polarization indices and names
        self.index_pol = {'E': 0, 'B': 1}
        self.pol_names = ['E', 'B']

    def check_sacc_consistency(self, s):
        """
        Checks the consistency of the SACC file and returns number of
        expected bandpowers.
        """
        bands = []
        splits = []
        for tn, t in s.tracers.items():
            if tn != 'temp':
                # If not template, tracer names are bandX_splitY
                band, split = tn.split('_', 2)
                bands.append(band)
                splits.append(split)
        bands = np.unique(bands)
        splits = np.unique(splits)
        if self.delensing:
            ntracers = self.nbands*self.nsplits+1
        else:
            ntracers = self.nbands*self.nsplits
        if ((len(bands) != self.nbands) or (len(splits) != self.nsplits) or
                (len(s.tracers) != ntracers)):
            raise ValueError("There's something wrong with these SACC tracers")
        if self.n_bpws == 0:
            self.ells, _ = s.get_ell_cl(s.data[0].data_type,
                                        s.data[0].tracers[0],
                                        s.data[0].tracers[1])
            self.n_bpws = len(self.ells)

        # Total number of power spectra expected
        nmaps = 2*ntracers
        nxt_expected = (ntracers * (ntracers + 1)) // 2
        nx_expected = (nmaps * (nmaps + 1)) // 2
        nv_expected = self.n_bpws*nx_expected
        if ((len(s.mean) != nv_expected) or
                (len(s.get_tracer_combinations()) != nxt_expected)):
            raise ValueError("There's something wrong with "
                             "the SACC data vector")

    def get_windows(self, s):
        self.windows = {}
        for b1 in range(self.n_channels):
            if b1 == self.n_channels_del-1:
                n1 = 'temp'
                nband1 = 'temp'
            else:
                n1 = 'band%d_split1' % (b1+1)
                nband1 = 'band%d' % (b1+1)
            for b2 in range(b1, self.n_channels):
                if b2 == self.n_channels_del-1:
                    n2 = 'temp'
                    nband2 = 'temp'
                else: # Only relevant case when delensing is off
                    n2 = 'band%d_split1' % (b2+1)
                    nband2 = 'band%d' % (b2+1)

                xname = nband1+'_'+nband2
                self.windows[xname] = {}
                _, _, ind = s.get_ell_cl('cl_ee', n1, n2, return_ind=True)
                self.windows[xname]['ee'] = s.get_bandpower_windows(ind)
                _, _, ind = s.get_ell_cl('cl_eb', n1, n2, return_ind=True)
                self.windows[xname]['eb'] = s.get_bandpower_windows(ind)
                self.windows[xname]['be'] = s.get_bandpower_windows(ind)
                _, _, ind = s.get_ell_cl('cl_bb', n1, n2, return_ind=True)
                self.windows[xname]['bb'] = s.get_bandpower_windows(ind)

    def get_tracers(self, s):
        """
        Gets two array of tracers: one for coadd SACC files,
        one for null SACC files.
        """
        tracers_bands = {}
        for tn, t in s.tracers.items():
            if tn != 'temp':
                band, split = tn.split('_', 2)
                if split == 'split1':
                    T = sacc.BaseTracer.make('NuMap', band,
                                             2, t.nu, t.bandpass,
                                             t.ell, t.beam,
                                             quantity='cmb_polarization',
                                             bandpass_extra={'dnu': t.bandpass_extra['dnu']})
                    tracers_bands[band] = T
            else:
                T = sacc.BaseTracer.make('Map', tn,
                                         2, t.ell, t.beam,
                                         quantity='cmb_polarization')
                tracer_temp = T

        self.t_coadd = []
        for i in range(self.nbands):
            self.t_coadd.append(tracers_bands['band%d' % (i+1)])
        if self.delensing:
            self.t_coadd.append(tracer_temp)

        self.t_nulls = []
        self.ind_nulls = {}
        ind_null = 0
        for b in range(self.nbands):
            t = tracers_bands['band%d' % (b+1)]
            # Loop over unique pairs
            for i in range(self.nsplits):
                for j in range(i, self.nsplits):
                    name = 'band%d_null%dm%d' % (b+1, i+1, j+1)
                    self.ind_nulls[name] = ind_null
                    T = sacc.BaseTracer.make('NuMap', name,
                                             2, t.nu, t.bandpass,
                                             t.ell, t.beam,
                                             quantity='cmb_polarization',
                                             bandpass_extra={'dnu': t.bandpass_extra['dnu']})
                    self.t_nulls.append(T)
                    ind_null += 1

    def bands_pol_iterator(self, half=True, with_windows=True):
        pols = ['e', 'b']
        for b1 in range(self.nbands):
            l1 = 'band%d' % (b1+1)
            if half:
                range_b2 = range(b1, self.nbands)
            else:
                range_b2 = range(self.nbands)
            for ip1 in range(2):
                for b2 in range_b2:
                    l2 = 'band%d' % (b2+1)
                    if (b1 == b2) and half:
                        p2_range = range(ip1, 2)
                    else:
                        p2_range = range(2)
                    for ip2 in p2_range:
                        x = pols[ip1] + pols[ip2]
                        if with_windows:
                            if b2 >= b1:
                                bname = 'band%d_band%d' % (b1+1, b2+1)
                                x_use = x
                            else:
                                bname = 'band%d_band%d' % (b2+1, b1+1)
                                x_use = x[::-1]
                            win = self.windows[bname][x_use]
                        else:
                            win = None
                        yield b1, ip1, b2, ip2, l1, l2, x, win

    def band_temp_pol_iterator(self, with_windows=True): # When delensing is on, this iterates over band x template spectra
        pols = ['e', 'b']
        for b1 in range(self.nbands):
            l1 = 'band%d' % (b1+1)
            b2 = self.n_channels_del-1
            l2 = 'temp'
            for ip1 in range(2):
                p2_range = range(2)
                for ip2 in p2_range:
                    x = pols[ip1] + pols[ip2]
                    if with_windows:
                        bname = l1+'_'+l2
                        x_use = x
                        win = self.windows[bname][x_use]
                    else:
                        win = None
                    yield b1, ip1, b2, ip2, l1, l2, x, win

    def temp_pol_iterator(self, half=True, with_windows=True): # When delensing is on, iterates over template x template spectra
        pols = ['e', 'b']
        b1 = self.n_channels_del-1
        l1 = 'temp'
        b2 = self.n_channels_del-1
        l2 = 'temp'
        for ip1 in range(2):
            if half:
                p2_range = range(ip1,2)
            else:
                p2_range = range(2)
            for ip2 in p2_range:
                x = pols[ip1] + pols[ip2]
                if with_windows:
                    bname = l1+'_'+l2
                    x_use = x
                    win = self.windows[bname][x_use]
                else:
                    win = None
                yield b1, ip1, b2, ip2, l1, l2, x, win

    def bands_splits_pol_iterator(self):
        for b1 in range(self.nbands):
            for b2 in range(b1, self.nbands):
                for s1 in range(self.nsplits):
                    if b1 == b2:
                        s2_range = range(s1, self.nsplits)
                    else:
                        s2_range = range(self.nsplits)
                    for s2 in s2_range:
                        for p1 in range(2):
                            if (s1 == s2) and (b1 == b2):
                                p2_range = range(p1, 2)
                            else:
                                p2_range = range(2)
                            for p2 in p2_range:
                                m1 = p1 + 2 * (b1 + self.nbands * s1)
                                m2 = p2 + 2 * (b2 + self.nbands * s2)
                                cl_name = ('cl_' + self.pol_names[p1].lower() +
                                           self.pol_names[p2].lower())
                                yield s1, s2, b1, b2, p1, p2, m1, m2, cl_name

    def band_temp_splits_pol_iterator(self): # Iterate over SAT splits x template cross-spectra
        for b1 in range(self.nbands):
            b2 = self.n_channels_del-1
            s2 = None
            for s1 in range(self.nsplits):
                for p1 in range(2):
                    p2_range = range(2)
                    for p2 in p2_range:
                        m1 = p1 + 2 * (b1 + self.nbands * s1)
                        m2 = p2
                        cl_name = ('cl_' + self.pol_names[p1].lower() +
                                    self.pol_names[p2].lower())
                        yield s1, s2, b1, b2, p1, p2, m1, m2, cl_name

    def temp_splits_pol_iterator(self): # Iterate over template spectra
        b1 = self.n_channels_del-1
        b2 = self.n_channels_del-1
        s1 = None
        s2 = None
        for p1 in range(2):
            p2_range = range(p1,2) # BE=EB
            for p2 in p2_range:
                m1 = p1
                m2 = p2
                cl_name = ('cl_' + self.pol_names[p1].lower() +
                            self.pol_names[p2].lower())
                yield s1, s2, b1, b2, p1, p2, m1, m2, cl_name

    def get_cl_indices(self, s):
        self.BxB_inds = np.zeros([(self.nsplits * self.nbands) * 2,
                              (self.nsplits * self.nbands) * 2,
                              self.n_bpws], dtype=int)
        itr1 = self.bands_splits_pol_iterator()
        for s1, s2, b1, b2, p1, p2, m1, m2, cltyp in itr1:
            t1 = 'band%d_split%d' % (b1+1, s1+1)
            t2 = 'band%d_split%d' % (b2+1, s2+1)
            _, _, ind = s.get_ell_cl(cltyp, t1, t2, return_ind=True)
            self.BxB_inds[m1, m2, :] = ind
            if m1 != m2:
                self.BxB_inds[m2, m1, :] = ind
        self.BxB_inds = self.BxB_inds.flatten()
        
        if self.delensing:
            self.BxTemp_inds = np.zeros([(self.nsplits * self.nbands) * 2, 2, self.n_bpws], dtype=int)
            itr2 = self.band_temp_splits_pol_iterator()
            for s1, s2, b1, b2, p1, p2, m1, m2, cltyp in itr2:
                t1 = 'band%d_split%d' % (b1+1, s1+1)
                t2 = 'temp'
                _, _, ind = s.get_ell_cl(cltyp, t1, t2, return_ind=True)
                self.BxTemp_inds[m1, m2, :] = ind
            self.BxTemp_inds = self.BxTemp_inds.flatten()

            self.TempxTemp_inds = np.zeros([2, 2, self.n_bpws], dtype=int)
            itr3 = self.temp_splits_pol_iterator()
            for s1, s2, b1, b2, p1, p2, m1, m2, cltyp in itr3:
                t1 = 'temp'
                t2 = 'temp'
                _, _, ind = s.get_ell_cl(cltyp, t1, t2, return_ind=True)
                self.TempxTemp_inds[m1, m2, :] = ind
                if m1 != m2:
                    self.TempxTemp_inds[m2, m1, :] = ind
            self.TempxTemp_inds = self.TempxTemp_inds.flatten()

    def parse_splits_sacc_file(self, s, get_saccs=True, with_windows=False):
        """
        Transform a SACC file containing splits into 4 SACC vectors:
        1 that contains the coadded power spectra.
        1 that contains coadded power spectra for cross-split only.
        1 that contains an estimate of the noise power spectrum.
        1 that contains all null tests
        """

        # Check we have the right number of bands, splits,
        # cross-correlations and power spectra
        self.check_sacc_consistency(s)

        # Now read power spectra into an array of form
        # [nsplits,nsplits,nbands,nbands,2,2,n_ell]
        # This duplicates the number of elements, but
        # simplifies bookkeeping significantly.

        # Put it in shape [nsplits,nsplits,nbands,2,nbands,2,nl]
        BxB_spectra = np.transpose(s.mean[self.BxB_inds].reshape([self.nsplits,
                                                          self.nbands, 2,
                                                          self.nsplits,
                                                          self.nbands, 2,
                                                          self.n_bpws]),
                               axes=[0, 3, 1, 2, 4, 5, 6])

        # Coadding (assuming flat coadding)
        # Total coadding (including diagonal)
        weights_total = np.ones(self.nsplits, dtype=float)/self.nsplits
        BxB_spectra_total = np.einsum('i,ijklmno,j',
                                  weights_total,
                                  BxB_spectra,
                                  weights_total)

        # Off-diagonal coadding
        triu_mean = np.mean(BxB_spectra[np.triu_indices(self.nsplits, 1)], axis=0)
        tril_mean = np.mean(BxB_spectra[np.tril_indices(self.nsplits, -1)], axis=0)
        BxB_spectra_xcorr = 0.5*(tril_mean+triu_mean)
        
        # Noise power spectra
        BxB_spectra_noise = BxB_spectra_total - BxB_spectra_xcorr

        if self.delensing:
            BxTemp_spectra = s.mean[self.BxTemp_inds].reshape([self.nsplits, self.nbands, 2, 2, self.n_bpws])
            BxTemp_spectra_total = np.einsum('i,ijklm', weights_total, BxTemp_spectra)
            BxTemp_spectra_xcorr = BxTemp_spectra_total
            BxTemp_spectra_noise = BxTemp_spectra_total - BxTemp_spectra_xcorr

            TempxTemp_spectra_total = s.mean[self.TempxTemp_inds].reshape([2, 2, self.n_bpws])
            TempxTemp_spectra_xcorr = TempxTemp_spectra_total
            TempxTemp_spectra_noise = TempxTemp_spectra_total - TempxTemp_spectra_xcorr

        # Nulls
        spectra_nulls = np.zeros([self.n_nulls,
                                  self.nbands, 2,
                                  self.nbands, 2,
                                  self.n_bpws])
        for i_null, (i, j, k, l) in enumerate(self.pairings):
            spectra_nulls[i_null] = (BxB_spectra[i, k]-BxB_spectra[i, l] -
                                     BxB_spectra[j, k]+BxB_spectra[j, l])

        ret = {}
        if get_saccs:

            s_total = sacc.Sacc()
            s_xcorr = sacc.Sacc()
            s_noise = sacc.Sacc()
            s_nulls = sacc.Sacc()
            for t in self.t_coadd:
                s_total.add_tracer_object(t)
                s_xcorr.add_tracer_object(t)
                s_noise.add_tracer_object(t)
            for t in self.t_nulls:
                s_nulls.add_tracer_object(t)

            itr = self.bands_pol_iterator(half=True,with_windows=with_windows)
            for b1, ip1, b2, ip2, l1, l2, x, win in itr:
                s_total.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   BxB_spectra_total[b1, ip1, b2, ip2],
                                   window=win)
                s_xcorr.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   BxB_spectra_xcorr[b1, ip1, b2, ip2],
                                   window=win)
                s_noise.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   BxB_spectra_noise[b1, ip1, b2, ip2],
                                   window=win)

            if self.delensing:
                itr2 = self.band_temp_pol_iterator(with_windows=with_windows)
                for b1, ip1, b2, ip2, l1, l2, x, win in itr2:
                    s_total.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       BxTemp_spectra_total[b1, ip1, ip2],
                                       window=win)
                    s_xcorr.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       BxTemp_spectra_xcorr[b1, ip1, ip2],
                                       window=win)
                    s_noise.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       BxTemp_spectra_noise[b1, ip1, ip2],
                                       window=win)

                itr3 = self.temp_pol_iterator(half=True, with_windows=with_windows)
                for b1, ip1, b2, ip2, l1, l2, x, win in itr3:
                    s_total.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       TempxTemp_spectra_total[ip1, ip2],
                                       window=win)
                    s_xcorr.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       TempxTemp_spectra_xcorr[ip1, ip2],
                                       window=win)
                    s_noise.add_ell_cl('cl_' + x, l1, l2,
                                       self.ells,
                                       TempxTemp_spectra_noise[ip1, ip2],
                                       window=win)

            for i_null, (i, j, k, l) in enumerate(self.pairings):
                itrb = self.bands_pol_iterator(half=False,
                                               with_windows=with_windows)
                for b1, ip1, b2, ip2, l1, l2, x, win in itrb:
                    l1s = l1 + '_null%dm%d' % (i+1, j+1)
                    l2s = l2 + '_null%dm%d' % (k+1, l+1)
                    s_nulls.add_ell_cl('cl_' + x, l1s, l2s,
                                       self.ells,
                                       spectra_nulls[i_null, b1, ip1, b2, ip2],
                                       window=win)
            ret['saccs'] = [s_total, s_xcorr, s_noise, s_nulls]

        # We use this to order the spectra correctly => the function only works if get_saccs = True => changed default to True
        spectra_total = s_total.mean
        spectra_xcorr = s_xcorr.mean
        spectra_noise = s_noise.mean
        spectra_nulls = spectra_nulls.reshape([-1, self.n_bpws]).flatten()

        ret['spectra'] = [spectra_total, spectra_xcorr, spectra_noise, spectra_nulls]

        return ret

    def run(self):
        # Set things up
        print("Init")
        self.init_params()

        # Create tracers for all future files
        print("Tracers")
        self.get_tracers(self.s_splits)

        print("Windows")
        self.get_windows(self.s_splits)

        print("Indexing")
        self.get_cl_indices(self.s_splits)

        # Read data file, coadd and compute nulls
        print("Reading data")
        summ = self.parse_splits_sacc_file(self.s_splits,
                                           get_saccs=True,
                                           with_windows=True)

        # Read simulations
        print("Reading simulations")
        sim_cd_t = np.zeros([self.nsims, len(summ['spectra'][0])])
        sim_cd_x = np.zeros([self.nsims, len(summ['spectra'][1])])
        sim_cd_n = np.zeros([self.nsims, len(summ['spectra'][2])])
        sim_null = np.zeros([self.nsims, len(summ['spectra'][3])])
        for i, fn in enumerate(self.fname_sims):
            print(fn)
            s = sacc.Sacc.load_fits(fn)
            sb = self.parse_splits_sacc_file(s)
            sim_cd_t[i, :] = sb['spectra'][0]
            sim_cd_x[i, :] = sb['spectra'][1]
            sim_cd_n[i, :] = sb['spectra'][2]
            sim_null[i, :] = sb['spectra'][3]

        sim_spectra_mean = np.mean(sim_cd_x,axis=0) # Average off-diagonal coadded spectra over all sims (useful to check that spectra aren't biased)

        # Compute covariance
        print("Covariances")
        dctyp = self.config['data_covar_type']
        dcord = self.config['data_covar_diag_order']
        self.get_covariance_from_samples(sim_cd_t, summ['saccs'][0],
                                         covar_type=dctyp,
                                         off_diagonal_cut=dcord)
        self.get_covariance_from_samples(sim_cd_x, summ['saccs'][1],
                                         covar_type=dctyp,
                                         off_diagonal_cut=dcord)
        self.get_covariance_from_samples(sim_cd_n, summ['saccs'][2],
                                         covar_type=dctyp,
                                         off_diagonal_cut=dcord)
        # There are so many nulls that we'll probably run out of memory
        nctyp = self.config['nulls_covar_type']
        ncord = self.config['nulls_covar_diag_order']
        self.get_covariance_from_samples(sim_null, summ['saccs'][3],
                                         covar_type=nctyp,
                                         off_diagonal_cut=ncord)

        # Save data
        print("Writing output")
        summ['saccs'][0].save_fits(self.get_output("cells_coadded_total"),
                                   overwrite=True)
        summ['saccs'][1].save_fits(self.get_output("cells_coadded"),
                                   overwrite=True)
        summ['saccs'][2].save_fits(self.get_output("cells_noise"),
                                   overwrite=True)
        summ['saccs'][3].save_fits(self.get_output("cells_null"),
                                   overwrite=True)

        print('Saving mean sims spectra')
        summ['saccs'][1].mean = sim_spectra_mean
        mean_dir = self.get_output("cells_coadded").rsplit('/',1)[0]
        print('Directory: ',mean_dir)
        summ['saccs'][1].save_fits(mean_dir+'/cells_mean_sims.fits', overwrite=True)

if __name__ == '__main_':
    cls = PipelineStage.main()

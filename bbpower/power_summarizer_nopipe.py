import numpy as np
import healpy as hp
import os
import argparse
import yaml
import time
import sacc
import bbpower.mpi_utils as mpi


def _yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


class BBPowerSummarizer(object):
    """
    Cross-bundle power spectrum coadder and empirical covariance stage
    for BB forecasts.
    """
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
        self.config = config["global"] | config["BBPowerSummarizer"]
        setattr(self, "config_fname", getattr(args, "config"))

        # Load global parameters
        self.nside = self.config['nside']
        self.npix = hp.nside2npix(self.nside)

        # Calculate number of bundles and number of frequency channels
        self.nbundles = len(self.config["bundle_ids"])
        self.nbands = len(list(self.config["map_sets"].keys()))

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
        for i in range(self.nbundles):
            # Loop over js that aren't i
            listj = list(filter(lambda x: x not in [i],
                                range(self.nbundles)))
            for j in listj:
                if j < i:
                    continue
                first_pairs.append((i, j))
                # ks that aren't j or i
                listk = list(filter(lambda x: x not in [i, j],
                                    range(self.nbundles)))
                for k in listk:
                    # l != i,j,k
                    listl = list(filter(lambda x: x not in [i, j, k],
                                        range(self.nbundles)))
                    for el in listl:
                        if el < k:
                            continue
                        if (k, el) in first_pairs:
                            continue
                        self.pairings.append((i, j, k, el))
        self.n_nulls = len(self.pairings)

        # First, initialize n_bpws to zero
        self.n_bpws = 0

        # Polarization indices and names
        self.index_pol = {'E': 0, 'B': 1}
        self.pol_names = ['E', 'B']

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

    def check_sacc_consistency(self, s):
        """
        Checks the consistency of the SACC file and returns number of
        expected bandpowers.
        """
        bands = []
        bundles = []
        for tn, t in s.tracers.items():
            # Tracer names are bandX_bundleY
            band, bundle = tn.split('_', 2)
            bands.append(band)
            bundles.append(bundle)
        bands = np.unique(bands)
        bundles = np.unique(bundles)
        if ((len(bands) != self.nbands) or (len(bundles) != self.nbundles) or
                (len(s.tracers) != self.nbands*self.nbundles)):
            raise ValueError("There's something wrong with these SACC tracers")
        if self.n_bpws == 0:
            self.ells, _ = s.get_ell_cl(s.data[0].data_type,
                                        s.data[0].tracers[0],
                                        s.data[0].tracers[1])
            self.n_bpws = len(self.ells)

        # Total number of power spectra expected
        ntracers = self.nbands * self.nbundles
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
        for b1 in range(self.nbands):
            n1 = 'band%d_bundle1' % (b1+1)
            for b2 in range(b1, self.nbands):
                n2 = 'band%d_bundle1' % (b2+1)
                xname = 'band%d_band%d' % (b1+1, b2+1)
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
            band, bundle = tn.split('_', 2)
            if bundle == 'bundle1':
                T = sacc.BaseTracer.make(
                    'NuMap', band, 2, t.nu, t.bandpass,
                    t.ell, t.beam, quantity='cmb_polarization',
                    bandpass_extra={'dnu': t.bandpass_extra['dnu']}
                )
                tracers_bands[band] = T

        self.t_coadd = []

        for i in range(self.nbands):
            self.t_coadd.append(tracers_bands['band%d' % (i+1)])

        self.t_nulls = []
        self.ind_nulls = {}
        ind_null = 0
        for b in range(self.nbands):
            t = tracers_bands['band%d' % (b+1)]
            # Loop over unique pairs
            for i in range(self.nbundles):
                for j in range(i, self.nbundles):
                    name = 'band%d_null%dm%d' % (b+1, i+1, j+1)
                    self.ind_nulls[name] = ind_null
                    T = sacc.BaseTracer.make(
                        'NuMap', name, 2, t.nu, t.bandpass,
                        t.ell, t.beam, quantity='cmb_polarization',
                        bandpass_extra={'dnu': t.bandpass_extra['dnu']}
                    )
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

    def bands_bundles_pol_iterator(self):
        for b1 in range(self.nbands):
            for b2 in range(b1, self.nbands):
                for s1 in range(self.nbundles):
                    if b1 == b2:
                        s2_range = range(s1, self.nbundles)
                    else:
                        s2_range = range(self.nbundles)
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

    def get_cl_indices(self, s):
        self.inds = np.zeros([self.nbundles * self.nbands * 2,
                              self.nbundles * self.nbands * 2,
                              self.n_bpws], dtype=int)

        itr = self.bands_bundles_pol_iterator()
        for s1, s2, b1, b2, _, _, m1, m2, cltyp in itr:
            t1 = 'band%d_bundle%d' % (b1+1, s1+1)
            t2 = 'band%d_bundle%d' % (b2+1, s2+1)
            _, _, ind = s.get_ell_cl(cltyp, t1, t2, return_ind=True)
            self.inds[m1, m2, :] = ind
            if m1 != m2:
                self.inds[m2, m1, :] = ind
        self.inds = self.inds.flatten()

    def parse_bundles_sacc_file(self, s, get_saccs=False, with_windows=False):
        """
        Transform a SACC file containing bundles into 4 SACC vectors:
        1 that contains the coadded power spectra.
        1 that contains coadded power spectra for cross-bundles only.
        1 that contains an estimate of the noise power spectrum.
        1 that contains all null tests
        """

        # Check we have the right number of bands, bundles,
        # cross-correlations and power spectra
        self.check_sacc_consistency(s)

        # Now read power spectra into an array of form
        # [nbundles,nbundles,nbands,nbands,2,2,n_ell]
        # This duplicates the number of elements, but
        # simplifies bookkeeping significantly.

        # Put it in shape [nbundles,nbundles,nbands,2,nbands,2,nl]
        spectra = np.transpose(s.mean[self.inds].reshape([self.nbundles,
                                                          self.nbands, 2,
                                                          self.nbundles,
                                                          self.nbands, 2,
                                                          self.n_bpws]),
                               axes=[0, 3, 1, 2, 4, 5, 6])

        # Coadding (assuming flat coadding)
        # Total coadding (including diagonal)
        weights_total = np.ones(self.nbundles, dtype=float)/self.nbundles
        spectra_total = np.einsum('i,ijklmno,j',
                                  weights_total,
                                  spectra,
                                  weights_total)

        # Off-diagonal coadding
        triu_mean = np.mean(spectra[np.triu_indices(self.nbundles, 1)], axis=0)
        tril_mean = np.mean(spectra[np.tril_indices(self.nbundles, -1)], axis=0)  # noqa
        spectra_xcorr = 0.5*(tril_mean+triu_mean)

        # Noise power spectra
        spectra_noise = spectra_total - spectra_xcorr

        # Nulls
        spectra_nulls = np.zeros([self.n_nulls,
                                  self.nbands, 2,
                                  self.nbands, 2,
                                  self.n_bpws])
        for i_null, (i, j, k, l) in enumerate(self.pairings):
            spectra_nulls[i_null] = (spectra[i, k]-spectra[i, l] -
                                     spectra[j, k]+spectra[j, l])

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

            itr = self.bands_pol_iterator(half=True,
                                          with_windows=with_windows)
            for b1, ip1, b2, ip2, l1, l2, x, win in itr:
                s_total.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   spectra_total[b1, ip1, b2, ip2],
                                   window=win)
                s_xcorr.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   spectra_xcorr[b1, ip1, b2, ip2],
                                   window=win)
                s_noise.add_ell_cl('cl_' + x, l1, l2,
                                   self.ells,
                                   spectra_noise[b1, ip1, b2, ip2],
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

        spectra_total = spectra_total.reshape([2*self.nbands, 2*self.nbands, self.n_bpws])[np.triu_indices(2*self.nbands)].flatten()  # noqa
        spectra_xcorr = spectra_xcorr.reshape([2*self.nbands, 2*self.nbands, self.n_bpws])[np.triu_indices(2*self.nbands)].flatten()  # noqa
        spectra_noise = spectra_noise.reshape([2*self.nbands, 2*self.nbands, self.n_bpws])[np.triu_indices(2*self.nbands)].flatten()  # noqa
        spectra_nulls = spectra_nulls.reshape([-1, self.n_bpws]).flatten()

        ret['spectra'] = [spectra_total,
                          spectra_xcorr,
                          spectra_noise,
                          spectra_nulls]
        return ret

    def run(self):
        """
        Run the cross-bundle power spectrum cooadder / covariance stage.
        """
        # Read cross-bundle power spectra
        cells_file = f"{self.config['cells_dir'].format(sim_id=self.sim_id)}/" + \
            f"{self.config['cells_format'].format(sim_id=self.sim_id)}"
        if not os.path.isfile(cells_file):
            print("FILE NOT FOUND. SKIPPING: \n"
                  f"  {cells_file}")
            return
        print(cells_file)
        self.s_bundles = sacc.Sacc.load_fits(cells_file)

        # Read sorting and number of bandpowers
        self.check_sacc_consistency(self.s_bundles)

        # Create tracers for all future files
        print("Tracers")
        self.get_tracers(self.s_bundles)

        print("Windows")
        self.get_windows(self.s_bundles)

        print("Indexing")
        self.get_cl_indices(self.s_bundles)

        # Read data file, coadd and compute nulls
        print("Reading data")
        summ = self.parse_bundles_sacc_file(self.s_bundles,
                                            get_saccs=True,
                                            with_windows=True)

        if self.do_covar:
            # Read simulations
            print("Reading simulations")
            cells_files = [
                f"{self.config['cells_dir'].format(sim_id=sim_id)}/"
                f"{self.config['cells_format'].format(sim_id=sim_id)}"
                for sim_id in self.sim_ids
            ]
            self.fname_sims = [x.strip()
                               for x in cells_files
                               if os.path.isfile(x.strip())]
            self.nsims = len(self.fname_sims)
            print(f"Found {self.nsims} sims on disk")

            sim_cd_t = np.zeros([self.nsims, len(summ['spectra'][0])])
            sim_cd_x = np.zeros([self.nsims, len(summ['spectra'][1])])
            sim_cd_n = np.zeros([self.nsims, len(summ['spectra'][2])])
            sim_null = np.zeros([self.nsims, len(summ['spectra'][3])])

            for i, fn in enumerate(self.fname_sims):
                print(fn)
                s = sacc.Sacc.load_fits(fn)
                sb = self.parse_bundles_sacc_file(s)
                sim_cd_t[i, :] = sb['spectra'][0]
                sim_cd_x[i, :] = sb['spectra'][1]
                sim_cd_n[i, :] = sb['spectra'][2]
                sim_null[i, :] = sb['spectra'][3]

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
            if self.config['do_null']:
                # There are so many nulls that we'll probably run out of memory
                nctyp = self.config['nulls_covar_type']
                ncord = self.config['nulls_covar_diag_order']
                self.get_covariance_from_samples(sim_null, summ['saccs'][3],
                                                 covar_type=nctyp,
                                                 off_diagonal_cut=ncord)

        # Save data
        print("Writing output")
        sacc_dir = self.config['sacc_dir'].format(sim_id=self.sim_id)
        sacc_file = f"{sacc_dir}/" + \
            f"{self.config['sacc_format'].format(sim_id=self.sim_id, typ=r'{typ}')}"  # noqa
        os.makedirs(sacc_dir, exist_ok=True)

        types = ["coadded_total", "coadded", "noise", "null"]
        for id_type, typ in enumerate(types):
            if f"do_{typ}" in self.config:
                if not self.config[f"do_{typ}"]:
                    continue
            summ['saccs'][id_type].save_fits(sacc_file.format(typ=typ),
                                             overwrite=True)


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
    rank, size, comm = mpi.init(switch=(not args.do_covar))
    mpi_shared_list = sim_ids if not args.do_covar else [0]

    # Every rank must have the same shared list
    if comm is not None:
        mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id in local_mpi_list:
        start = time.time()
        summarizer = BBPowerSummarizer(args)
        setattr(summarizer, "sim_id", sim_id)
        setattr(summarizer, "sim_ids", sim_ids)
        setattr(summarizer, "do_covar", args.do_covar)
        summarizer.run()
    mpi.print_rnk0(f"Processed {len(sim_ids)} simulations "
                   f"in {time.time() - start:.1f} seconds.", rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cross-bundle spectrum coadder and covariance stage "
                    "for BB forecasting"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to yaml file with pipeline configuration"
    )
    parser.add_argument(
        "--do_covar", action="store_true",
        help="Compute power spectrum covariance"
    )

    args = parser.parse_args()
    main(args)

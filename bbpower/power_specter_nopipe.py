import numpy as np
import healpy as hp
import pymaster as nmt
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


class BBPowerSpecter(object):
    """
    Cross-bundle power spectrum computation stage for BB forecasts.
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
        self.config = config["global"] | config["BBPowerSpecter"]
        setattr(self, "config_fname", getattr(args, "config"))

        # Load global parameters
        self.nside = self.config['nside']
        self.npix = hp.nside2npix(self.nside)

    def read_beams(self, nbeams):
        """
        """
        from scipy.interpolate import interp1d

        beam_fnames = [map_set["beam_file"]
                       for map_set in self.config["map_sets"].values()]

        # Check that there are enough beams
        if len(beam_fnames) != nbeams:
            raise ValueError("Couldn't find enough beams: "
                             f"({len(beam_fnames)} != {nbeams})")

        self.larr_all = np.arange(3*self.nside)
        self.beams = {}
        for i_f, f in enumerate(beam_fnames):
            li, bi = np.loadtxt(f, unpack=True)
            bb = interp1d(li, bi, fill_value=0,
                          bounds_error=False)(self.larr_all)
            if li[0] != 0:
                bb[:int(li[0])] = bi[0]
            self.beams['band%d' % (i_f+1)] = bb

    def compute_cells_from_bundles(self, bundles_list):
        """
        """
        print(" Generating fields")
        has_tqu = True
        ftest = bundles_list[0]
        if not os.path.isfile(ftest):
            ftest = ftest + '.gz'
        try:
            hp.read_map(ftest, field=[0])
        except:  # noqa
            raise ValueError(f'Map is not readable: {ftest}')
        try:
            hp.read_map(ftest, field=[17])
        except:  # noqa
            has_tqu = False

        fields = {}
        for b in range(self.n_bpss):
            for s in range(self.nbundles):
                name = self.get_map_label(b, s)
                print("  "+name)
                fname = bundles_list[s]
                if not os.path.isfile(fname):
                    fname = fname + '.gz'
                if not os.path.isfile(fname):
                    raise ValueError("Can't find file ", bundles_list[s])
                if has_tqu:
                    mp_q, mp_u = hp.read_map(fname, field=[3*b+1, 3*b+2])
                else:
                    mp_q, mp_u = hp.read_map(fname, field=[2*b, 2*b+1])
                fields[name] = self.get_field(b, [mp_q, mp_u])

        # Iterate over field pairs
        print(" Computing cross-spectra")
        cells = {}
        for b1, b2, _, _, l1, l2 in self.get_cell_iterator():
            wsp = self.workspaces[self.get_workspace_label(b1, b2)]
            # Create sub-dictionary if it doesn't exist
            if cells.get(l1) is None:
                cells[l1] = {}
            f1 = fields[l1]
            f2 = fields[l2]
            # Compute power spectrum
            print("  "+l1+" "+l2)
            cells[l1][l2] = wsp.decouple_cell(nmt.compute_coupled_cell(f1, f2))

        return cells

    def read_bandpasses(self):
        """
        """
        bpss_fnames = [map_set["bandpass_file"]
                       for map_set in self.config["map_sets"].values()]
        self.n_bpss = len(bpss_fnames)

        self.bpss = {}
        for i_f, f in enumerate(bpss_fnames):
            nu, bnu = np.loadtxt(f, unpack=True)
            dnu = np.zeros_like(nu)
            dnu[1:] = np.diff(nu)
            dnu[0] = dnu[1]
            self.bpss['band%d' % (i_f+1)] = {'nu': nu,
                                             'dnu': dnu,
                                             'bnu': bnu}

    def read_masks(self, nbands):
        """
        Assume a single mask for each map set (to avoid mask-based
        decorrelation during compsep).
        """
        self.masks = []
        for i in range(nbands):
            # Always load same mask (keep self.masks a list for generality)
            m = hp.read_map(self.config["mask_path"])
            self.masks.append(hp.ud_grade(m, nside_out=self.nside))

    def get_bandpowers(self):
        """
        """
        # If it's a file containing the bandpower edges
        if isinstance(self.config['bpw_edges'], str):
            # Custom spacing
            edges = np.loadtxt(self.config['bpw_edges']).astype(int)
            ells = np.arange(3*self.nside)
            bpws = np.zeros(3*self.nside, dtype=int)-1
            weights = np.ones(3*self.nside)

            for ibpw, (l0, lf) in enumerate(zip(edges[:-1], edges[1:])):
                if lf < 3*self.nside:
                    bpws[l0:lf] = ibpw
            # Add more equi-spaced bandpowers up to the end of the band
            if edges[-1] < 3*self.nside:
                dell = edges[-1]-edges[-2]
                l0 = edges[-1]
                while l0+dell < 3*self.nside:
                    ibpw += 1
                    bpws[l0:l0+dell] = ibpw
                    l0 += dell
            self.bins = nmt.NmtBin(
                bpws=bpws,
                ells=self.larr_all,
                weights=weights,
                f_ell=ells*(ells+1)/(2*np.pi)
            )
        else:  # otherwise it could be a constant integer interval
            self.bins = nmt.NmtBin.from_nside_linear(
                self.nside,
                nlb=int(self.config['bpw_edges']),
                is_Dell=True
            )

    def get_fname_workspace(self, band1, band2):
        """
        """
        b1 = min(band1, band2)
        b2 = max(band1, band2)
        return self.config["mcm_path"].format(band1=b1, band2=b2)

    def get_field(self, band, mps, spin=2):
        """
        """
        f = nmt.NmtField(self.masks[band],
                         mps,
                         beam=self.beams['band%d' % (band+1)],
                         purify_b=self.config['purify_B'],
                         n_iter=self.config['n_iter'],
                         spin=spin)
        return f

    def compute_workspace(self, band1, band2):
        """
        """
        b1 = min(band1, band2)
        b2 = max(band1, band2)

        w = nmt.NmtWorkspace()
        fname = self.get_fname_workspace(b1+1, b2+1)

        # If file exists, just read it
        if os.path.isfile(fname):
            print("Reading %d %d" % (b1+1, b2+1))
            w.read_from(fname)
        elif self.do_mpi:
            raise ValueError(
                "No mode coupling matrices found. Please run "
                "BBPowerSpecter with --do_mcm to compute MCMs."
            )
        else:
            print("Computing %d %d" % (b1+1, b2+1))
            f1 = self.get_field(b1, None, spin=2)
            f2 = self.get_field(b2, None, spin=2)
            w.compute_coupling_matrix(f1, f2, self.bins)
            w.write_to(fname)

        return w

    def get_map_label(self, band, bundle):
        """
        """
        return 'band%d_bundle%d' % (band+1,  bundle+1)

    def get_workspace_label(self, band1, band2):
        """
        """
        b1 = min(band1, band2)
        b2 = max(band1, band2)
        return 'b%d_b%d' % (b1+1, b2+1)

    def compute_workspaces(self):
        """
        Compute MCMs for all possible band combinations.
        Assumption is that mask and beams are different across bands,
        but the same across polarization channels and bundles.
        """
        print("Estimating mode-coupling matrices")
        self.workspaces = {}
        for i1 in range(self.n_bpss):
            for i2 in range(i1, self.n_bpss):
                name = self.get_workspace_label(i1, i2)
                self.workspaces[name] = self.compute_workspace(i1, i2)

    def get_cell_iterator(self):
        """
        """
        for c1 in range(self.n_bpss):
            for c2 in range(c1, self.n_bpss):
                for b1 in range(self.nbundles):
                    l1 = self.get_map_label(c1, b1)
                    if c1 == c2:
                        bundles_range = range(b1, self.nbundles)
                    else:
                        bundles_range = range(self.nbundles)
                    for b2 in bundles_range:
                        l2 = self.get_map_label(c2, b2)
                        yield (c1, c2, b1, b2, l1, l2)

    def get_sacc_tracers(self):
        """
        """
        sacc_t = []
        for c in range(self.n_bpss):
            bpss = self.bpss['band%d' % (c+1)]
            beam = self.beams['band%d' % (c+1)]
            for b in range(self.nbundles):
                T = sacc.BaseTracer.make('NuMap', self.get_map_label(c, b),
                                         2, bpss['nu'], bpss['bnu'],
                                         self.larr_all, beam,
                                         quantity='cmb_polarization',
                                         bandpass_extra={'dnu': bpss['dnu']})
                sacc_t.append(T)
        return sacc_t

    def get_sacc_windows(self):
        """
        """
        windows_wsp = {}
        for c1 in range(self.n_bpss):
            for c2 in range(c1, self.n_bpss):
                name = self.get_workspace_label(c1, c2)
                windows_wsp[name] = {}
                wsp = self.workspaces[name]
                bpw_win = wsp.get_bandpower_windows()
                windows_wsp[name]['EE'] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[0, :, 0, :].T
                )
                windows_wsp[name]['EB'] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[1, :, 1, :].T
                )
                windows_wsp[name]['BE'] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[2, :, 2, :].T
                )
                windows_wsp[name]['BB'] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[3, :, 3, :].T
                )
        return windows_wsp

    def save_cell_to_file(self, cell, tracers, fname, with_windows=False):
        """
        """
        # Create sacc file
        s = sacc.Sacc()

        # Add tracers
        for t in tracers:
            s.add_tracer_object(t)

        # Add each power spectrum
        l_eff = self.bins.get_effective_ells()
        for c1, c2, b1, b2, l1, l2 in self.get_cell_iterator():
            add_BE = not ((c1 == c2) and (b1 == b2))
            if with_windows:
                wname = self.get_workspace_label(c1, c2)
                s.add_ell_cl(
                    'cl_ee', l1, l2, l_eff, cell[l1][l2][0],
                    window=self.win[wname]['EE']
                )
                s.add_ell_cl(
                    'cl_eb', l1, l2, l_eff, cell[l1][l2][1],
                    window=self.win[wname]['EB']
                )
                if add_BE:  # Only add B1E2 if 1!=2
                    s.add_ell_cl(
                        'cl_be', l1, l2, l_eff, cell[l1][l2][2],
                        window=self.win[wname]['BE']
                    )
                s.add_ell_cl(
                    'cl_bb', l1, l2, l_eff, cell[l1][l2][3],
                    window=self.win[wname]['BB']
                )
            else:
                s.add_ell_cl('cl_ee', l1, l2, l_eff, cell[l1][l2][0])
                s.add_ell_cl('cl_eb', l1, l2, l_eff, cell[l1][l2][1])
                if add_BE:  # Only add B1E2 if 1!=2
                    s.add_ell_cl('cl_be', l1, l2, l_eff, cell[l1][l2][2])
                s.add_ell_cl('cl_bb', l1, l2, l_eff, cell[l1][l2][3])

        print("Saving to ", fname)
        s = s.save_fits(fname, overwrite=True)

    def run(self):
        """
        Run the cross-bundle power spectrum computation stage.
        """
        # Read bandpasses
        print("Reading bandpasses")
        self.read_bandpasses()

        # Read beams
        print("Reading beams")
        self.read_beams(self.n_bpss)

        # Create bandpowers
        self.get_bandpowers()

        # Read masks
        print("Reading masks")
        self.read_masks(self.n_bpss)

        # Compute all possible MCMs
        self.compute_workspaces()

        # Compile list of bundles
        self.nbundles = len(self.config["bundle_ids"])
        map_file = f"{self.config['map_dir']}/" \
            + self.config['map_format'].format(sim_id=self.sim_id,
                                               n_bundles=self.nbundles,
                                               bundle_id=r"{bundle_id}")
        self.bundles = [
            map_file.format(bundle_id=bundle_id)
            for bundle_id in self.config["bundle_ids"]
        ]
        if not os.path.isfile(self.bundles[0]):
            print(
                f"WARNING: maps for sim {self.sim_id:04d} do not exist. "
                "Skipping."
            )
            return

        # Get SACC binning
        self.win = self.get_sacc_windows()

        # Get SACC tracers
        self.tracers = self.get_sacc_tracers()

        # Compute all possible cross-power spectra
        print("Computing all cross-correlations")
        cells = self.compute_cells_from_bundles(self.bundles)

        # Save output
        print("Saving to file")
        cells_file = f"{self.config['cells_dir']}/" + \
            f"{self.config['cells_format'].format(sim_id=self.sim_id)}"
        self.save_cell_to_file(cells,
                               self.tracers,
                               cells_file,
                               with_windows=True)


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
    rank, size, comm = mpi.init(switch=(not args.do_mcm))
    mpi_shared_list = sim_ids if not args.do_mcm else [0]

    # Every rank must have the same shared list
    if comm is not None:
        mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id in local_mpi_list:
        start = time.time()
        specter = BBPowerSpecter(args)
        setattr(specter, "sim_id", sim_id)
        setattr(specter, "do_mpi", size > 1)
        specter.run()
    mpi.print_rnk0(f"Processed {len(sim_ids)} simulations "
                   f"in {time.time() - start:.1f} seconds.", rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cross-bundle spectra for BB forecasting"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to yaml file with pipeline configuration"
    )
    parser.add_argument(
        "--do_mcm", action="store_true",
        help="Compute mode coupling matrices"
    )

    args = parser.parse_args()
    main(args)

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bbpipe import PipelineStage
from .types import FitsFile, TextFile, DummyFile
import sacc
import numpy as np
import healpy as hp
import pymaster as nmt
import inspect
import os


class BBPowerSpecter(PipelineStage):
    """
    Compute cross-frequency/split/polarization power spectra from HEALPix maps.

    Uses NaMaster (pymaster) to estimate pseudo-C_l power spectra with
    purified B-modes. Reads bandpasses, beams, and apodized masks, computes
    mode-coupling matrices for all frequency-band pairs, and produces
    decoupled EE/EB/BE/BB bandpowers saved in SACC format. Also processes
    simulation splits to build an ensemble of Monte Carlo spectra.
    """

    name = "BBPowerSpecter"
    inputs = [
        ("splits_list", TextFile),
        ("masks_apodized", FitsFile),
        ("bandpasses_list", TextFile),
        ("sims_list", TextFile),
        ("beams_list", TextFile),
    ]
    outputs = [
        ("cells_all_splits", FitsFile),
        ("cells_all_sims", TextFile),
        ("mcm", DummyFile),
    ]
    config_options = {"bpw_edges": None, "purify_B": True, "n_iter": 3}

    def init_params(self) -> None:
        """
        Initialize basic parameters from the pipeline configuration.

        Sets ``nside``, ``npix``, and the mode-coupling matrix file prefix
        used by downstream methods.
        """
        self.nside = self.config["nside"]
        self.npix = hp.nside2npix(self.nside)
        self.prefix_mcm = self.get_output("mcm")[:-4]

    def read_beams(self, nbeams: int) -> None:
        """
        Read beam transfer functions and interpolate onto the ell array.

        Reads beam files listed in the ``beams_list`` input, interpolates
        each beam onto ``self.larr_all`` (0 to 3*nside-1), and stores the
        results in ``self.beams``.

        Parameters
        ----------
        nbeams : int
            Expected number of beam files (must match the number of
            frequency bands).

        Raises
        ------
        ValueError
            If the number of beam files does not equal ``nbeams``.
        """
        from scipy.interpolate import interp1d

        beam_fnames = []
        with open(self.get_input("beams_list"), "r") as f:
            for fname in f:
                beam_fnames.append(fname.strip())

        # Check that there are enough beams
        if len(beam_fnames) != nbeams:
            raise ValueError(
                "Couldn't find enough beams: " f"{len(beam_fnames)} != {nbeams}"
            )

        self.larr_all = np.arange(3 * self.nside)
        self.beams = {}
        for i_f, f in enumerate(beam_fnames):
            li, bi = np.loadtxt(f, unpack=True)
            bb = interp1d(li, bi, fill_value=0, bounds_error=False)(self.larr_all)
            if li[0] != 0:
                bb[: int(li[0])] = bi[0]
            self.beams[f"band{i_f+1}"] = bb

    def compute_cells_from_splits(self, splits_list: list[str]) -> dict:
        """
        Compute all cross-power spectra from a list of split map files.

        Creates NaMaster fields for every (band, split) combination by
        reading Q/U maps from the provided files, then computes decoupled
        pseudo-C_l cross-spectra for all unique field pairs.

        Parameters
        ----------
        splits_list : list of str
            Paths to HEALPix FITS files, one per split. Each file must
            contain 2*n_bpss maps (Q and U for each frequency band).

        Returns
        -------
        cells : dict of dict
            Nested dictionary keyed by ``(label1, label2)`` map labels,
            where each value is an array of shape ``(4, n_ell)``
            containing the EE, EB, BE, BB decoupled bandpowers.
        """
        # Generate fields
        print(" Generating fields")
        fields = {}
        for b in range(self.n_bpss):
            for s in range(self.nsplits):
                name = self.get_map_label(b, s)
                print("  " + name)
                fname = splits_list[s]
                if not os.path.isfile(fname):  # See if it's gzipped
                    fname = fname + ".gz"
                if not os.path.isfile(fname):
                    raise ValueError(f"Can't find file {splits_list[s]}")
                mp_q, mp_u = hp.read_map(fname, field=[2 * b, 2 * b + 1])
                fields[name] = self.get_field(b, [mp_q, mp_u])

        # Iterate over field pairs
        print(" Computing cross-spectra")
        cells = {}
        for b1, b2, s1, s2, l1, l2 in self.get_cell_iterator():
            wsp = self.workspaces[self.get_workspace_label(b1, b2)]
            # Create sub-dictionary if it doesn't exist
            if cells.get(l1) is None:
                cells[l1] = {}
            f1 = fields[l1]
            f2 = fields[l2]
            # Compute power spectrum
            print("  " + l1 + " " + l2)
            cells[l1][l2] = wsp.decouple_cell(nmt.compute_coupled_cell(f1, f2))

        return cells

    def read_bandpasses(self) -> None:
        """
        Read bandpass profiles from the files listed in ``bandpasses_list``.

        Populates ``self.bpss`` with one entry per frequency band, each
        containing arrays for frequency (``nu``), frequency spacing
        (``dnu``), and bandpass response (``bnu``). Also sets
        ``self.n_bpss`` to the number of bands.
        """
        bpss_fnames = []
        with open(self.get_input("bandpasses_list"), "r") as f:
            for fname in f:
                bpss_fnames.append(fname.strip())
        self.n_bpss = len(bpss_fnames)
        self.bpss = {}
        for i_f, f in enumerate(bpss_fnames):
            nu, bnu = np.loadtxt(f, unpack=True)
            dnu = np.zeros_like(nu)
            dnu[1:] = np.diff(nu)
            dnu[0] = dnu[1]
            self.bpss[f"band{i_f+1}"] = {"nu": nu, "dnu": dnu, "bnu": bnu}

    def read_masks(self, nbands: int) -> None:
        """
        Read the apodized mask and replicate it for each frequency band.

        The mask is re-graded to the working ``nside`` and stored in
        ``self.masks``.

        Parameters
        ----------
        nbands : int
            Number of frequency bands; one copy of the mask is stored
            per band.
        """
        self.masks = []
        for i in range(nbands):
            m = hp.read_map(self.get_input("masks_apodized"))
            self.masks.append(hp.ud_grade(m, nside_out=self.nside))

    @staticmethod
    def _nmt_bin_uses_keyword_api() -> bool:
        """Check whether ``NmtBin`` uses the NaMaster 2 keyword-only API.

        Returns
        -------
        bool
            True when the installed NaMaster exposes the 2.x constructor
            signature ``NmtBin(*, bpws, ells, ..., f_ell=...)``. False for
            the older positional constructor used by NaMaster 1.x.
        """
        try:
            params = inspect.signature(nmt.NmtBin).parameters
        except (TypeError, ValueError):
            return False
        return "f_ell" in params and "is_Dell" not in params and "nside" not in params

    @staticmethod
    def _dell_prefactor(ells: np.ndarray) -> np.ndarray:
        """Return the multiplicative factor that converts ``C_ell`` to ``D_ell``.

        Parameters
        ----------
        ells : numpy.ndarray
            Multipoles at which the prefactor should be evaluated.

        Returns
        -------
        numpy.ndarray
            The factor ``ell * (ell + 1) / (2 * pi)``.
        """
        return ells * (ells + 1) / (2 * np.pi)

    def _make_custom_nmt_bin(
        self,
        bpws: np.ndarray,
        weights: np.ndarray,
        is_dell: bool,
    ) -> nmt.NmtBin:
        """Create a custom NaMaster bin object across NaMaster 1.x and 2.x.

        Parameters
        ----------
        bpws : numpy.ndarray
            Bandpower index assigned to each multipole. Negative values are
            ignored by NaMaster.
        weights : numpy.ndarray
            Per-multipole weights for the bandpower averages.
        is_dell : bool
            If True, make decoupled outputs use ``D_ell`` units instead of
            ``C_ell`` units.

        Returns
        -------
        pymaster.NmtBin
            Binning scheme compatible with the installed NaMaster version.
        """
        if self._nmt_bin_uses_keyword_api():
            # NaMaster 2 removed the old is_Dell keyword from the low-level
            # constructor. Passing f_ell preserves the historical behavior.
            f_ell = self._dell_prefactor(self.larr_all) if is_dell else None
            return nmt.NmtBin(
                bpws=bpws,
                ells=self.larr_all,
                weights=weights,
                f_ell=f_ell,
            )
        return nmt.NmtBin(
            self.nside,
            bpws=bpws,
            ells=self.larr_all,
            weights=weights,
            is_Dell=is_dell,
        )

    def _make_linear_nmt_bin(self, nlb: int) -> nmt.NmtBin:
        """Create a linear NaMaster bin object across NaMaster 1.x and 2.x.

        Parameters
        ----------
        nlb : int
            Constant bandpower width in multipoles.

        Returns
        -------
        pymaster.NmtBin
            Linear binning scheme compatible with the installed NaMaster
            version.
        """
        if self._nmt_bin_uses_keyword_api():
            return nmt.NmtBin.from_nside_linear(self.nside, nlb)
        return nmt.NmtBin(self.nside, nlb=nlb)

    @staticmethod
    def _compute_coupling_matrix(
        workspace: nmt.NmtWorkspace,
        field_1: nmt.NmtField,
        field_2: nmt.NmtField,
        bins: nmt.NmtBin,
        n_iter: int,
    ) -> None:
        """Compute a coupling matrix across NaMaster 1.x and 2.x.

        Parameters
        ----------
        workspace : pymaster.NmtWorkspace
            Workspace object to populate.
        field_1, field_2 : pymaster.NmtField
            Fields whose mode-coupling matrix should be computed.
        bins : pymaster.NmtBin
            Bandpower binning scheme.
        n_iter : int
            Spherical harmonic iteration count. NaMaster 1.x accepted this
            on ``compute_coupling_matrix``; NaMaster 2.x takes it on
            ``NmtField`` instead, so it must not be passed twice.
        """
        params = inspect.signature(workspace.compute_coupling_matrix).parameters
        if "n_iter" in params:
            # NaMaster 1 accepted n_iter here; keep passing it for old installs.
            workspace.compute_coupling_matrix(
                field_1,
                field_2,
                bins,
                n_iter=n_iter,
            )
            return
        # NaMaster 2 moved n_iter to NmtField and rejects it on workspaces.
        workspace.compute_coupling_matrix(field_1, field_2, bins)

    def get_bandpowers(self) -> None:
        """
        Set up NaMaster bandpower binning from the configuration.

        If ``bpw_edges`` is a filename, the edges are read from that file
        and extended with equal-width bins to 3*nside. If ``bpw_edges`` is
        an integer, uniform bins of that width are used. The resulting
        ``NmtBin`` object is stored as ``self.bins``.
        """
        # If it's a file containing the bandpower edges
        if isinstance(self.config["bpw_edges"], str):
            # Custom spacing
            edges = np.loadtxt(self.config["bpw_edges"]).astype(int)
            bpws = np.zeros(3 * self.nside, dtype=int) - 1
            weights = np.ones(3 * self.nside)
            for ibpw, (l0, lf) in enumerate(zip(edges[:-1], edges[1:])):
                if lf < 3 * self.nside:
                    bpws[l0:lf] = ibpw
            # Add more equi-spaced bandpowers up to the end of the band
            if edges[-1] < 3 * self.nside:
                dell = edges[-1] - edges[-2]
                l0 = edges[-1]
                while l0 + dell < 3 * self.nside:
                    ibpw += 1
                    bpws[l0 : l0 + dell] = ibpw
                    l0 += dell

            is_dell = False
            if self.config.get("compute_dell"):
                is_dell = True
            self.bins = self._make_custom_nmt_bin(bpws, weights, is_dell)
        else:  # otherwise it could be a constant integer interval
            self.bins = self._make_linear_nmt_bin(int(self.config["bpw_edges"]))

    def get_fname_workspace(self, band1: int, band2: int) -> str:
        """
        Return the FITS filename for a mode-coupling matrix workspace.

        Parameters
        ----------
        band1, band2 : int
            Zero-based frequency band indices (order does not matter).

        Returns
        -------
        str
            Path of the form ``<prefix_mcm>_<b1>_<b2>.fits``.
        """
        b1 = min(band1, band2)
        b2 = max(band1, band2)
        return f"{self.prefix_mcm}_{b1+1}_{b2+1}.fits"

    def get_field(self, band: int, mps: list) -> Any:
        """
        Create an NaMaster spin-2 field with the appropriate mask and beam.

        Parameters
        ----------
        band : int
            Zero-based frequency band index, used to select the mask
            and beam.
        mps : list of array_like
            Two HEALPix maps ``[Q, U]`` for the polarization field.

        Returns
        -------
        pymaster.NmtField
            NaMaster field configured with B-mode purification and the
            iteration count from the pipeline configuration.
        """
        f = nmt.NmtField(
            self.masks[band],
            mps,
            beam=self.beams[f"band{band+1}"],
            purify_b=self.config["purify_B"],
            n_iter=self.config["n_iter"],
        )
        return f

    def compute_workspace(self, band1: int, band2: int) -> Any:
        """
        Compute or load the mode-coupling matrix for a band pair.

        If a pre-computed workspace FITS file already exists on disk it is
        read; otherwise the MCM is computed from dummy fields and saved.

        Parameters
        ----------
        band1, band2 : int
            Zero-based frequency band indices.

        Returns
        -------
        pymaster.NmtWorkspace
            The mode-coupling matrix workspace for the given band pair.
        """
        b1 = min(band1, band2)
        b2 = max(band1, band2)

        w = nmt.NmtWorkspace()
        fname = self.get_fname_workspace(b1, b2)
        # If file exists, just read it
        if os.path.isfile(fname):
            print(f"Reading {b1} {b2}")
            w.read_from(fname)
        else:
            print(f"Computing {b1} {b2}")
            mdum = np.zeros([2, self.npix])
            f1 = self.get_field(b1, mdum)
            f2 = self.get_field(b2, mdum)
            self._compute_coupling_matrix(w, f1, f2, self.bins, self.config["n_iter"])
            w.write_to(fname)

        return w

    def get_map_label(self, band: int, split: int) -> str:
        """Return the SACC tracer name for a (band, split) pair."""
        return f"band{band+1}_split{split+1}"

    def get_workspace_label(self, band1: int, band2: int) -> str:
        """Return the canonical workspace key for a band pair (order-independent)."""
        b1 = min(band1, band2)
        b2 = max(band1, band2)
        return f"b{b1+1}_b{b2+1}"

    def compute_workspaces(self) -> None:
        """
        Compute mode-coupling matrices for all unique band pairs.

        Iterates over the upper triangle of band combinations (including
        the diagonal) and stores the resulting workspaces in
        ``self.workspaces``, keyed by workspace label strings.
        """
        # Compute MCMs for all possible band combinations.
        #  Assumption is that mask is different across bands,
        #  but the same across polarization channels and splits.
        print("Estimating mode-coupling matrices")
        self.workspaces = {}
        for i1 in range(self.n_bpss):
            for i2 in range(i1, self.n_bpss):
                name = self.get_workspace_label(i1, i2)
                self.workspaces[name] = self.compute_workspace(i1, i2)

    def get_cell_iterator(self) -> Iterator[tuple[int, int, int, int, str, str]]:
        """
        Yield all unique (band, split) cross-pair combinations.

        Iterates over the upper triangle of band pairs and, for each,
        over the appropriate split pairs (upper triangle when bands are
        equal, full matrix otherwise).

        Yields
        ------
        b1, b2 : int
            Zero-based band indices.
        s1, s2 : int
            Zero-based split indices.
        l1, l2 : str
            Map labels for the two fields (e.g. ``'band1_split1'``).
        """
        for b1 in range(self.n_bpss):
            for b2 in range(b1, self.n_bpss):
                for s1 in range(self.nsplits):
                    l1 = self.get_map_label(b1, s1)
                    if b1 == b2:
                        splits_range = range(s1, self.nsplits)
                    else:
                        splits_range = range(self.nsplits)
                    for s2 in splits_range:
                        l2 = self.get_map_label(b2, s2)
                        yield (b1, b2, s1, s2, l1, l2)

    def get_sacc_tracers(self) -> list[Any]:
        """
        Create SACC tracer objects for all band/split combinations.

        Each tracer is a ``NuMap`` tracer carrying the bandpass, beam,
        and CMB polarization metadata for one (band, split) pair.

        Returns
        -------
        list of sacc.BaseTracer
            One tracer per (band, split) combination, ordered by band
            then split.
        """
        sacc_t = []
        for b in range(self.n_bpss):
            bpss = self.bpss[f"band{b+1}"]
            beam = self.beams[f"band{b+1}"]
            for s in range(self.nsplits):
                T = sacc.BaseTracer.make(
                    "NuMap",
                    self.get_map_label(b, s),
                    2,
                    bpss["nu"],
                    bpss["bnu"],
                    self.larr_all,
                    beam,
                    quantity="cmb_polarization",
                    bandpass_extra={"dnu": bpss["dnu"]},
                )
                sacc_t.append(T)
        return sacc_t

    def get_sacc_windows(self) -> dict[str, dict[str, Any]]:
        """
        Extract bandpower window functions from all workspaces.

        Builds SACC ``BandpowerWindow`` objects for the EE, EB, BE, and
        BB spectra of each unique band pair.

        Returns
        -------
        dict
            Nested dictionary ``{workspace_label: {pol: BandpowerWindow}}``
            where ``pol`` is one of ``'EE'``, ``'EB'``, ``'BE'``, ``'BB'``.
        """
        windows_wsp = {}
        for b1 in range(self.n_bpss):
            for b2 in range(b1, self.n_bpss):
                name = self.get_workspace_label(b1, b2)
                windows_wsp[name] = {}
                wsp = self.workspaces[name]
                bpw_win = wsp.get_bandpower_windows()
                windows_wsp[name]["EE"] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[0, :, 0, :].T
                )
                windows_wsp[name]["EB"] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[1, :, 1, :].T
                )
                windows_wsp[name]["BE"] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[2, :, 2, :].T
                )
                windows_wsp[name]["BB"] = sacc.BandpowerWindow(
                    self.larr_all, bpw_win[3, :, 3, :].T
                )
        return windows_wsp

    def save_cell_to_file(
        self, cell: dict, tracers: list, fname: str, with_windows: bool = False
    ) -> None:
        """
        Save power spectra and tracers to a SACC FITS file.

        Writes EE, EB, (optionally BE), and BB bandpowers for every
        cross-pair produced by ``get_cell_iterator``. The BE spectrum is
        omitted for auto-spectra (same band and split) because it is
        identical to EB by symmetry.

        Parameters
        ----------
        cell : dict of dict
            Nested dictionary of decoupled spectra, as returned by
            ``compute_cells_from_splits``.
        tracers : list of sacc.BaseTracer
            SACC tracer objects to include in the output file.
        fname : str
            Output FITS file path.
        with_windows : bool, optional
            If True, attach bandpower window functions to each spectrum
            entry. Default is False.
        """
        # Create sacc file
        s = sacc.Sacc()

        # Add tracers
        for t in tracers:
            s.add_tracer_object(t)

        # Add each power spectrum
        l_eff = self.bins.get_effective_ells()
        for b1, b2, s1, s2, l1, l2 in self.get_cell_iterator():
            add_BE = not ((b1 == b2) and (s1 == s2))
            if with_windows:
                wname = self.get_workspace_label(b1, b2)
                s.add_ell_cl(
                    "cl_ee",
                    l1,
                    l2,
                    l_eff,
                    cell[l1][l2][0],
                    window=self.win[wname]["EE"],
                )  # EE
                s.add_ell_cl(
                    "cl_eb",
                    l1,
                    l2,
                    l_eff,
                    cell[l1][l2][1],
                    window=self.win[wname]["EB"],
                )  # EB
                if add_BE:  # Only add B1E2 if 1!=2
                    s.add_ell_cl(
                        "cl_be",
                        l1,
                        l2,
                        l_eff,
                        cell[l1][l2][2],
                        window=self.win[wname]["BE"],
                    )  # BE
                s.add_ell_cl(
                    "cl_bb",
                    l1,
                    l2,
                    l_eff,
                    cell[l1][l2][3],
                    window=self.win[wname]["BB"],
                )  # EE
            else:
                s.add_ell_cl("cl_ee", l1, l2, l_eff, cell[l1][l2][0])  # EE
                s.add_ell_cl("cl_eb", l1, l2, l_eff, cell[l1][l2][1])  # EB
                if add_BE:  # Only add B1E2 if 1!=2
                    s.add_ell_cl("cl_be", l1, l2, l_eff, cell[l1][l2][2])  # BE
                s.add_ell_cl("cl_bb", l1, l2, l_eff, cell[l1][l2][3])  # EE

        print("Saving to " + fname)
        s = s.save_fits(fname, overwrite=True)

    def run(self) -> None:
        """
        Execute the full power spectrum pipeline.

        Sequentially reads bandpasses, beams, and masks; sets up bandpower
        binning; computes mode-coupling matrices; measures cross-spectra
        for the data splits; and then processes each simulation directory.
        Data spectra are saved with bandpower windows; simulation spectra
        are saved without windows. Existing simulation output files are
        skipped.
        """
        self.init_params()

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

        # Compile list of splits
        splits = []
        with open(self.get_input("splits_list"), "r") as f:
            for fname in f:
                splits.append(fname.strip())
        self.nsplits = len(splits)

        # Get SACC binning
        self.win = self.get_sacc_windows()

        # Get SACC tracers
        self.tracers = self.get_sacc_tracers()

        # Compute all possible cross-power spectra
        print("Computing all cross-correlations")
        cell_data = self.compute_cells_from_splits(splits)

        # Save output
        print("Saving to file")
        self.save_cell_to_file(
            cell_data,
            self.tracers,
            self.get_output("cells_all_splits"),
            with_windows=True,
        )
        # Iterate over simulations
        sims = []
        with open(self.get_input("sims_list"), "r") as f:
            for dname in f:
                sims.append(dname.strip())

        # Write all output file names into a text file
        fo = open(self.get_output("cells_all_sims"), "w")
        prefix_out = self.get_output("cells_all_splits")[:-5]
        for isim, d in enumerate(sims):
            fname = f"{prefix_out}_sim{isim}.fits"
            fo.write(fname + "\n")
        fo.close()

        for isim, d in enumerate(sims):
            fname = f"{prefix_out}_sim{isim}.fits"
            if os.path.isfile(fname):
                print("found " + fname)
                continue
            print(f"{isim+1}-th / {len(sims)} simulation")
            #   Compute list of splits
            sim_splits = [
                f"{d}/obs_split{i+1}of{self.nsplits}.fits" for i in range(self.nsplits)
            ]
            #   Compute all possible cross-power spectra
            cell_sim = self.compute_cells_from_splits(sim_splits)
            #   Save output
            fname = f"{prefix_out}_sim{isim}.fits"
            self.save_cell_to_file(cell_sim, self.tracers, fname, with_windows=False)


if __name__ == "__main__":
    cls = PipelineStage.main()

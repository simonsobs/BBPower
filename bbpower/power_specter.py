from bbpipe import PipelineStage
from .types import FitsFile,TextFile,DummyFile
import sacc
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class BBPowerSpecter(PipelineStage):
    """
    Template for a power spectrum stage
    """
    name="BBPowerSpecter"
    inputs=[('splits_list',TextFile),('masks_apodized',FitsFile),('bandpasses_list',TextFile),
            ('sims_list',TextFile),('beams_list',TextFile)]
    outputs=[('cells_all_splits',FitsFile),('cells_all_sims',TextFile),('mcm',DummyFile)]
    config_options={'bpw_edges':None,
                    'beam_correct':True,
                    'purify_B':True,
                    'n_iter':3}

    def init_params(self):
        self.nside = self.config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.prefix_mcm = self.get_output('mcm')[:-4]

    def read_beams(self,nbeams):
        from scipy.interpolate import interp1d

        beam_fnames = []
        with open(self.get_input('beams_list'),'r') as f:
            for fname in f:
                beam_fnames.append(fname.strip())

        # Check that there are enough beams
        if len(beam_fnames)!=nbeams:
            raise ValueError("Couldn't find enough beams %d != %d" (len(beam_fnames),nbeams))

        self.larr_all = np.arange(3*self.nside)
        self.beams={}
        for i_f,f in enumerate(beam_fnames):
            li,bi=np.loadtxt(f,unpack=True)
            bb=interp1d(li,bi,fill_value=0,bounds_error=False)(self.larr_all)
            if li[0]!=0:
                bb[:int(li[0])]=bi[0]
            self.beams['band%d' % (i_f+1)]=bb

    def compute_cells_from_splits(self, splits_list):
        # Generate fields
        print(" Generating fields")
        fields = {}
        for b in range(self.n_bpss):
            for s in range(self.nsplits):
                name = self.get_map_label(b,s)
                print("  "+name)
                fname = splits_list[s]
                if not os.path.isfile(fname):  # See if it's gzipped
                    fname = fname + '.gz'
                if not os.path.isfile(fname):
                    raise ValueError("Can't find file ",splits_list[s])
                mp_q,mp_u=hp.read_map(fname, field=[2*b,2*b+1], verbose=False)
                fields[name] = self.get_field(b,[mp_q,mp_u])

        # Iterate over field pairs
        print(" Computing cross-spectra")
        cells = {}
        for b1,b2,s1,s2,l1,l2 in self.get_cell_iterator():
            wsp = self.workspaces[self.get_workspace_label(b1,b2)]
            if cells.get(l1) is None: # Create sub-dictionary if it doesn't exist
                cells[l1]={}
            f1 = fields[l1]
            f2 = fields[l2]
            # Compute power spectrum
            print("  "+l1+" "+l2)
            cells[l1][l2] = wsp.decouple_cell(nmt.compute_coupled_cell(f1,f2))

        return cells

    def read_bandpasses(self):
        bpss_fnames = []
        with open(self.get_input('bandpasses_list'),'r') as f:
            for fname in f:
                bpss_fnames.append(fname.strip())
        self.n_bpss = len(bpss_fnames)
        self.bpss={}
        for i_f,f in enumerate(bpss_fnames):
            nu,bnu=np.loadtxt(f,unpack=True)
            dnu=np.zeros_like(nu)
            dnu[1:]=np.diff(nu)
            dnu[0]=dnu[1]
            self.bpss['band%d' % (i_f+1)]={'nu':nu, 'dnu':dnu, 'bnu':bnu}

    def read_masks(self,nbands):
        self.masks=[]
        for i in range(nbands):
            m=hp.read_map(self.get_input('masks_apodized'),
                          verbose=False)
            self.masks.append(hp.ud_grade(m,nside_out=self.nside))

    def get_bandpowers(self):
        # If it's a file containing the bandpower edges
        if isinstance(self.config['bpw_edges'],str):
            # Custom spacing
            edges = np.loadtxt(self.config['bpw_edges']).astype(int)
            bpws = np.zeros(3*self.nside,dtype=int)-1
            weights = np.ones(3*self.nside)
            for ibpw,(l0,lf) in enumerate(zip(edges[:-1],edges[1:])):
                if lf<3*self.nside:
                    bpws[l0:lf]=ibpw
            # Add more equi-spaced bandpowers up to the end of the band
            if edges[-1]<3*self.nside:
                dell = edges[-1]-edges[-2]
                l0 = edges[-1]
                while l0+dell<3*self.nside:
                    ibpw+=1
                    bpws[l0:l0+dell]=ibpw
                    l0+=dell

            is_dell = False
            if self.config.get('compute_dell'):
                is_dell = True
            self.bins=nmt.NmtBin(self.nside,
                                 bpws=bpws,
                                 ells=self.larr_all,
                                 weights=weights,
                                 is_Dell=is_dell)
        else: # otherwise it could be a constant integer interval
            self.bins=nmt.NmtBin(self.nside,nlb=int(self.config['bpw_edges']))

    def get_fname_workspace(self,band1,band2):
        b1=min(band1,band2)
        b2=max(band1,band2)
        return self.prefix_mcm+"_%d_%d.fits" % (b1+1, b2+1)

    def get_field(self,band,mps):
        f = nmt.NmtField(self.masks[band],
                         mps,
                         beam=self.beams['band%d' % (band+1)],
                         purify_b=self.config['purify_B'],
                         n_iter=self.config['n_iter'])
        return f

    def compute_workspace(self,band1,band2):
        b1=min(band1,band2)
        b2=max(band1,band2)

        w = nmt.NmtWorkspace()
        fname = self.get_fname_workspace(b1,b2)
        # If file exists, just read it
        if os.path.isfile(fname):
            print("Reading %d %d" % (b1, b2))
            w.read_from(fname)
        else:
            print("Computing %d %d" % (b1, b2))
            mdum = np.zeros([2,self.npix])
            f1=self.get_field(b1,mdum)
            f2=self.get_field(b2,mdum)
            w.compute_coupling_matrix(f1,f2,self.bins,n_iter=self.config['n_iter'])
            w.write_to(fname)

        return w

    def get_map_label(self,band,split):
        return 'band%d_split%d' % (band+1, split+1)

    def get_workspace_label(self,band1,band2):
        b1=min(band1,band2)
        b2=max(band1,band2)
        return 'b%d_b%d' % (b1+1, b2+1)

    def compute_workspaces(self):
        # Compute MCMs for all possible band combinations.
        #  Assumption is that mask is different across bands,
        #  but the same across polarization channels and splits.
        print("Estimating mode-coupling matrices")
        self.workspaces={}
        for i1 in range(self.n_bpss):
            for i2 in range(i1,self.n_bpss):
                name=self.get_workspace_label(i1,i2)
                self.workspaces[name] = self.compute_workspace(i1,i2)

    def get_cell_iterator(self):
        for b1 in range(self.n_bpss):
            for b2 in range(b1,self.n_bpss):
                for s1 in range(self.nsplits):
                    l1=self.get_map_label(b1,s1)
                    if b1==b2:
                        splits_range=range(s1,self.nsplits)
                    else:
                        splits_range=range(self.nsplits)
                    for s2 in splits_range:
                        l2=self.get_map_label(b2,s2)
                        yield(b1,b2,s1,s2,l1,l2)

    def get_sacc_tracers(self):
        sacc_t = []
        for b in range(self.n_bpss):
            bpss = self.bpss['band%d' % (b+1)]
            beam = self.beams['band%d' % (b+1)]
            for s in range(self.nsplits):
                T = sacc.BaseTracer.make('NuMap', self.get_map_label(b, s),
                                         2, bpss['nu'], bpss['bnu'],
                                         self.larr_all, beam,
                                         quantity='cmb_polarization',
                                         bandpass_extra={'dnu': bpss['dnu']})
                sacc_t.append(T)
        return sacc_t
                                          
    def get_sacc_windows(self):
        windows_wsp = {}
        for b1 in range(self.n_bpss):
            for b2 in range(b1,self.n_bpss):
                name = self.get_workspace_label(b1, b2)
                windows_wsp[name]={}
                wsp = self.workspaces[name]
                bpw_win = wsp.get_bandpower_windows()
                windows_wsp[name]['EE'] = sacc.BandpowerWindow(self.larr_all, bpw_win[0, :, 0, :].T)
                windows_wsp[name]['EB'] = sacc.BandpowerWindow(self.larr_all, bpw_win[1, :, 1, :].T)
                windows_wsp[name]['BE'] = sacc.BandpowerWindow(self.larr_all, bpw_win[2, :, 2, :].T)
                windows_wsp[name]['BB'] = sacc.BandpowerWindow(self.larr_all, bpw_win[3, :, 3, :].T)
        return windows_wsp

    def save_cell_to_file(self, cell, tracers, fname, with_windows=False):
        # Create sacc file
        s = sacc.Sacc()

        # Add tracers
        for t in tracers:
            s.add_tracer_object(t)

        # Add each power spectrum
        l_eff = self.bins.get_effective_ells()
        for b1,b2,s1,s2,l1,l2 in self.get_cell_iterator():
            add_BE = not ((b1==b2) and (s1==s2))
            if with_windows:
                wname = self.get_workspace_label(b1, b2)
                s.add_ell_cl('cl_ee', l1, l2, l_eff, cell[l1][l2][0],
                             window=self.win[wname]['EE'])  # EE
                s.add_ell_cl('cl_eb', l1, l2, l_eff, cell[l1][l2][1],
                             window=self.win[wname]['EB'])  # EB
                if add_BE: #Only add B1E2 if 1!=2
                    s.add_ell_cl('cl_be', l1, l2, l_eff, cell[l1][l2][2],
                                 window=self.win[wname]['BE'])  # BE
                s.add_ell_cl('cl_bb', l1, l2, l_eff, cell[l1][l2][3],
                             window=self.win[wname]['BB'])  # EE
            else:
                s.add_ell_cl('cl_ee', l1, l2, l_eff, cell[l1][l2][0])  # EE
                s.add_ell_cl('cl_eb', l1, l2, l_eff, cell[l1][l2][1])  # EB
                if add_BE: #Only add B1E2 if 1!=2
                    s.add_ell_cl('cl_be', l1, l2, l_eff, cell[l1][l2][2])  # BE
                s.add_ell_cl('cl_bb', l1, l2, l_eff, cell[l1][l2][3])  # EE

        print("Saving to "+fname)
        s = s.save_fits(fname, overwrite=True)

    def run(self) :
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
        with open(self.get_input('splits_list'),'r') as f:
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
        self.save_cell_to_file(cell_data,
                               self.tracers,
                               self.get_output('cells_all_splits'),
                               with_windows=True)
        # Iterate over simulations
        sims = []
        with open(self.get_input('sims_list'),'r') as f:
            for dname in f:
                sims.append(dname.strip())

        # Write all output file names into a text file
        fo=open(self.get_output('cells_all_sims'),'w')
        prefix_out=self.get_output('cells_all_splits')[:-5]
        for isim,d in enumerate(sims):
            fname=prefix_out + "_sim%d.fits" % isim
            fo.write(fname+"\n")
        fo.close()

        for isim,d in enumerate(sims):
            fname=prefix_out + "_sim%d.fits" % isim
            if os.path.isfile(fname):
                print("found " + fname)
                continue
            print("%d-th / %d simulation" % (isim+1, len(sims)))
            #   Compute list of splits
            sim_splits = [d+'/obs_split%dof%d.fits' % (i+1, self.nsplits)
                          for i in range(self.nsplits)]
            #   Compute all possible cross-power spectra
            cell_sim=self.compute_cells_from_splits(sim_splits)
            #   Save output
            fname=prefix_out + "_sim%d.fits" % isim
            self.save_cell_to_file(cell_sim,
                                   self.tracers,
                                   fname, with_windows=False)

if __name__ == '__main__':
    cls = PipelineStage.main()

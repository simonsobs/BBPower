import numpy as np
import matplotlib.pyplot as plt

from bbpipe import PipelineStage
from .types import NpzFile, FitsFile, YamlFile, DirFile
from .param_manager import ParameterManager


class BBEllwise(PipelineStage):
    """
    CMB bandpower stage
    This stage saves CMB bandpowers after ellwise component separation.
    """
    name = "BBEllwise"
    inputs = [('default_chain', DirFile)]
    outputs = [('output_dir', DirFile)]
    config_options = {'sampler': 'emcee', 'validation': True}

    def bin_cells(self, bin_edges, ells, cells):
        """
        Bins power spectra into bandpower windows, given bin edges.
        bin_edges contains [lo_1, lo_2, lo_3, lo_4, ...,lo_n, hi_n+1],
        where n-th bin has range [lo_n, hi_n=lo_n+1 - 1].
        """
        cells_b = np.zeros(len(bin_edges)-1)
        for i_b in range(len(bin_edges)-1):
            target = bin_edges[i_b] + np.diff(bin_edges)[i_b]/2
            value = cells[np.where(ells==target)]
            if value:
                cells_b[i_b] = value[-1]
        assert(len(cells_b) == len(bin_edges)-1)
        return np.asarray(cells_b)

    def read_ell_mc(self, mcfile):
        """
        Reads CMB bandpowers from emcee output.
        Assumes bandpowers = first nell parameters.
        """
        mcfile = np.load(mcfile)
        pnames = mcfile['names']
        chain = mcfile['chain']
        A = np.zeros(self.nell)
        A_std = np.zeros(self.nell)
        for il in range(self.nell):
            pname = f'A_cmb_{str(il+1).zfill(2)}'
            assert(pname == pnames[il])
            A[il] = np.mean(chain[:,:,il], axis=(0,1))
            A_std[il] = np.std(chain[:,:,il], axis=(0,1))

        return A, A_std

    def read_ell_pc(self, pcfile):
        """
        Reads CMB bandpowers from PolyChord output.
        Assumes bandpowers = first nell parameters.
        """
        A = np.zeros(self.nell)
        A_std = np.zeros(self.nell)
        with open(pcfile) as f:
            for line in f:
                if line.startswith("Dim No."):
                    break
            for il in range(self.nell):
                l = f.readline()
                A[il] += float(l.split()[1])
                A_std[il] += float(l.split()[3])

        return A, A_std

    def read_r_AL_mc(self, mc_npz_dir):
        """
        Reads r, A_lens from emcee output.
        Assumes A_lens, r are first two parameters.
        """
        mc_chain = np.load(mc_npz_dir)['chain']
        AL_fid = np.mean(mc_chain[:,:,0], axis=(0,1))
        AL_std_fid = np.std(mc_chain[:,:,0], axis=(0,1))
        r_fid = np.mean(mc_chain[:,:,1], axis=(0,1))
        r_std_fid = np.std(mc_chain[:,:,1], axis=(0,1))

        return r_fid, r_std_fid, AL_fid, AL_std_fid

    def read_r_AL_pc(self, pch_stats_dir):
        """
        Reads r, A_lens from PolyChord output.
        Assumes A_lens, r are first two parameters.
        """
        with open(pch_stats_dir) as fid:
            for line in fid:
                if line.startswith("Dim No."):
                    break
            l = fid.readline() 
            AL_fid = float(l.split()[1])
            AL_std_fid = float(l.split()[3])
            l = fid.readline() 
            r_fid = float(l.split()[1])
            r_std_fid = float(l.split()[3])

        return r_fid, r_std_fid, AL_fid, AL_std_fid

    def validate(self, A, A_std):
        """
        Inserts CMB bandpowers and standard errors (in units of
        lensing B-modes, assuming no correlations) in likelihood
        parameterized by (r, A_lens), and outputs marginal posteriors.
        """
        import emcee, os
        from multiprocessing import Pool

        def lnprob_cmb(theta):
            res = (A - theta[1])*self.cmb_lens_b - theta[0]*self.cmb_tens_b
            return -0.5*np.sum(np.square(res)/self.cmb_lens_b**2 / A_std**2)

        def emcee_sampler_valid():
            fname_temp = self.get_output('output_dir')+'/valid_ellwise.h5'
            backend = emcee.backends.HDFBackend(fname_temp)

            nwalkers = 5
            n_iters = 1000
            ndim = 2
            found_file = os.path.isfile(fname_temp)

            try:
                nchain = len(backend.get_chain())
            except AttributeError:
                found_file = False

            if not found_file:
                backend.reset(nwalkers, ndim)
                pos = [np.asarray([0.01,1.]) + 1.e-3*np.random.randn(ndim)
                       for i in range(nwalkers)]
                nsteps_use = n_iters
            else:
                print("Restarting from previous run")
                pos = None
                nsteps_use = max(n_iters - nchain, 0)

            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                lnprob_cmb,
                                                backend=backend)
                if nsteps_use > 0:
                    sampler.run_mcmc(pos, nsteps_use, store=True, progress=False)

            return sampler

        print('Validating CMB bandpowers')
        import time
        start = time.time()
        sampler_valid = emcee_sampler_valid()
        end = time.time()
        print(f" Finished validation in {(end-start):.0f} s")

        chain_valid = sampler_valid.chain
        r_valid = np.mean(chain_valid[:,:,0], axis=(0,1))
        r_std_valid = np.std(chain_valid[:,:,0], axis=(0,1))
        AL_valid = np.mean(chain_valid[:,:,1], axis=(0,1))
        AL_std_valid = np.std(chain_valid[:,:,1], axis=(0,1))

        return r_valid, r_std_valid, AL_valid, AL_std_valid

    def run(self):
        # Load bandpower metadata
        bin_edges = np.loadtxt(self.config.get('bin_edges_path'))
        lmin = self.config.get('lmin')
        lmax = self.config.get('lmax')
        ell_mask = (bin_edges>lmin) * (bin_edges<lmax+3)
        bin_edges = bin_edges[np.where(ell_mask)]
        dell = np.mean(np.diff(bin_edges))

        # Load fiducial power spectra
        cmb_ells = np.loadtxt(self.config.get('fid_spec_r1'))[:,0]
        mask = (cmb_ells <= lmax+3) & (cmb_ells > max(1,lmin))
        cmb_ells = cmb_ells[mask]
        cmb_dl2cl = 2*np.pi/cmb_ells/(cmb_ells+1)
        cmb_r1 = np.loadtxt(self.config.get('fid_spec_r1'))[:,3][mask]
        cmb_tens = (np.loadtxt(self.config.get('fid_spec_r1'))[:,3] - \
                    np.loadtxt(self.config.get('fid_spec_r0'))[:,3])[mask]
        cmb_lens = np.loadtxt(self.config.get('fid_spec_r0'))[:,3][mask]
        self.cmb_tens_b = self.bin_cells(bin_edges, cmb_ells, cmb_tens)
        self.cmb_lens_b = self.bin_cells(bin_edges, cmb_ells, cmb_lens)
        ells_b = self.bin_cells(bin_edges, cmb_ells, cmb_ells)
        self.nell = len(ells_b)

        # Load CMB bandpowers
        if self.config.get('sampler') == 'emcee':
            A, A_std = self.read_ell_mc(self.get_output('output_dir')+'/emcee_ellwise.npz')
        elif self.config.get('sampler') == 'polychord':
            A, A_std = self.read_ell_pc(self.get_output('output_dir')+'/polychord_ellwise/pch.stats')
        else:
            raise ValueError('Unknown sampler')
        print('CMB bandpowers, best fit in units of input CMB:\n', A)
        np.savez(self.get_output('output_dir')+'/cl_cmb.npz', ells=ells_b, 
                 dells=self.cmb_lens_b*A, errs=self.cmb_lens_b*A_std)
        print('Saved under "'+self.get_output('output_dir')+'/cl_cmb.npz"')

        
        if self.config.get('validation') == True:
            # Load marginal validated posteriors on r and A_lens
            r, r_std, AL, AL_std = self.validate(A, A_std)

            # Load marginal fiducial posteriors on r and A_lens
            if '.npz' in self.get_input('default_chain'):
                r_fid, r_std_fid, AL_fid, AL_std_fid = self.read_r_AL_mc(self.get_input('default_chain'))
            elif '.stats' in self.get_input('default_chain'):
                r_fid, r_std_fid, AL_fid, AL_std_fid = self.read_r_AL_pc(self.get_input('default_chain'))
            else:
                print('Default chain file has the wrong format - skipping.')

            print('*** Validation: ***')
            print(f'r      = {r:.5f} +/- {r_std:.5f} from CMB bandpowers')
            try:
                print(f'r      = {r_fid:.5f} +/- {r_std_fid:.5f} from default likelihood')
            except:
                pass
            print(f'A_lens = {AL:.3f} +/- {AL_std:.3f} from CMB bandpowers')
            try:
                print(f'A_lens = {AL_fid:.3f} +/- {AL_std_fid:.3f}  from default likelihood')
            except:
                pass
        
        return
    
    
if __name__ == '__main__':
    cls = PipelineStage.main()
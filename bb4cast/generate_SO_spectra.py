import numpy as np
from utils import *
import noise_calc as nc
import sacc
import os
import sys
import yaml


def get_spectra(info):
    fname_coadd = info['prefix_out'] + "/cls_coadd.fits"
    fname_fid = info['prefix_out'] + "/cls_fid.fits"
    fname_noise = info['prefix_out'] + "/cls_noise.fits"
    recompute = info.get('recompute_data', True)
    if (os.path.isfile(fname_coadd) and
        os.path.isfile(fname_fid) and
        os.path.isfile(fname_noise) and
        (not recompute)):
        return

    # Bandpasses
    bpss = {n: Bpass(n, info['bands'][n].get('bandpass',
                                             f'examples/data/bandpasses/{n}.txt'))
            for n in band_names}

    # Bandpowers
    dell = 10
    nbands = 100
    lmax = 2+nbands*dell
    ls = np.arange(lmax+1)
    lbands = np.linspace(2,lmax,nbands+1,dtype=int)
    leff = 0.5*(lbands[1:]+lbands[:-1])
    windows = np.zeros([nbands,lmax+1])
    cl2dl=ls*(ls+1)/(2*np.pi)
    dl2cl=np.zeros_like(cl2dl)
    dl2cl[1:] = 1/(cl2dl[1:])
    for b,(l0,lf) in enumerate(zip(lbands[:-1],lbands[1:])):
        windows[b,l0:lf] = cl2dl[l0:lf]
        windows[b,:] /= dell
    s_wins = sacc.BandpowerWindow(ls, windows.T)

    # Beams
    beams = {bn: np.exp(-0.5*ls*(ls+1)*fwhm2s(info['bands'][bn]['beam'])**2)
             for bn in band_names}

    # Component spectra
    dls_comp=np.zeros([3,2,3,2,lmax+1]) #[ncomp,np,ncomp,np,nl]
    (dls_comp[1,0,1,0,:],
     dls_comp[1,1,1,1,:],
     dls_comp[2,0,2,0,:],
     dls_comp[2,1,2,1,:],
     dls_comp[0,0,0,0,:],
     dls_comp[0,1,0,1,:]) = get_component_spectra(info, lmax)
    dls_comp *= dl2cl[None, None, None, None, :]

    # Convolve with windows
    bpw_comp=np.sum(dls_comp[:,:,:,:,None,:]*windows[None,None,None,None,:,:],axis=5)

    # Convolved SEDs
    seds = get_convolved_seds(info, band_names, bpss)
    _, nfreqs = seds.shape

    # Component -> frequencies
    bpw_freq_sig=np.einsum('ik,jm,iljno',seds,seds,bpw_comp)

    # N_ell
    _, nell = nc.so_SAT_Nl(info, lmax)
    for ib, b in enumerate(band_names):
        refac = np.ones_like(beams[b])
        goodl = beams[b] > 1E-50
        refac[goodl] = 1./beams[b][goodl]**2
        refac[~goodl] = 1E100
        nell[ib] *= refac
    n_bpw=np.sum(nell[:,None,:]*windows[None,:,:],axis=2)
    bpw_freq_noi=np.zeros_like(bpw_freq_sig)
    for ib,n in enumerate(n_bpw):
        bpw_freq_noi[ib,0,ib,0,:]=n_bpw[ib,:]
        bpw_freq_noi[ib,1,ib,1,:]=n_bpw[ib,:]

    # Add to signal
    bpw_freq_tot=bpw_freq_sig+bpw_freq_noi
    bpw_freq_tot=bpw_freq_tot.reshape([nfreqs*2,nfreqs*2,nbands])
    bpw_freq_sig=bpw_freq_sig.reshape([nfreqs*2,nfreqs*2,nbands])
    bpw_freq_noi=bpw_freq_noi.reshape([nfreqs*2,nfreqs*2,nbands])

    # Creating Sacc files
    s_d = sacc.Sacc()
    s_f = sacc.Sacc()
    s_n = sacc.Sacc()

    # Adding tracers
    for ib, n in enumerate(band_names):
        bandpass = bpss[n]
        beam = beams[n]
        for s in [s_d, s_f, s_n]:
            s.add_tracer('NuMap', 'band%d' % (ib+1),
                         quantity='cmb_polarization',
                         spin=2,
                         nu=bandpass.nu,
                         bandpass=bandpass.bnu,
                         ell=ls,
                         beam=beam,
                         nu_unit='GHz',
                         map_unit='uK_CMB')

    # Adding power spectra
    nmaps=2*nfreqs
    ncross=(nmaps*(nmaps+1))//2
    indices_tr=np.triu_indices(nmaps)
    map_names=[]
    for ib, n in enumerate(band_names):
        map_names.append('band%d' % (ib+1) + '_E')
        map_names.append('band%d' % (ib+1) + '_B')
    for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
        n1 = map_names[i1][:-2]
        n2 = map_names[i2][:-2]
        p1 = map_names[i1][-1].lower()
        p2 = map_names[i2][-1].lower()
        cl_type = f'cl_{p1}{p2}'
        s_d.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
        s_f.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
        s_n.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_noi[i1, i2, :], window=s_wins)

    # Add covariance
    fsky = info.get('f_sky', 0.1)
    cov_bpw = np.zeros([ncross, nbands, ncross, nbands])
    factor_modecount = 1./((2*leff+1)*dell*fsky)
    for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
        for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            covar = (bpw_freq_tot[i1, j1, :]*bpw_freq_tot[i2, j2, :]+
                     bpw_freq_tot[i1, j2, :]*bpw_freq_tot[i2, j1, :]) * factor_modecount
            cov_bpw[ii, :, jj, :] = np.diag(covar)
    cov_bpw = cov_bpw.reshape([ncross * nbands, ncross * nbands])
    s_d.add_covariance(cov_bpw)

    # Write output
    s_d.save_fits(fname_coadd, overwrite=True)
    s_f.save_fits(fname_fid, overwrite=True)
    s_n.save_fits(fname_noise, overwrite=True)


def get_bbpower_command(info):
    prefix = info['prefix_out']+'/'
    pyexec = info.get('pyexec', 'python')
    cmd = f'{pyexec} -m bbpower BBCompSep '
    cmd += f'--cells_coadded={prefix}cls_coadd.fits '
    cmd += f'--cells_noise={prefix}cls_noise.fits '
    cmd += f'--cells_fiducial={prefix}cls_fid.fits '
    cmd += f'--param_chains={prefix}param_chains.npz '
    cmd += f'--config_copy={prefix}bbpower_copy '
    cmd += f'--config={prefix}config.yml'
    return cmd


def get_bbpower_config(info):
    A_sync_BB = info['fg'].get('A_sync_BB', 2.0)
    EB_sync = info['fg'].get('EB_sync', 2.0)
    alpha_sync_EE = info['fg'].get('alpha_sync_EE', -0.6)
    alpha_sync_BB = info['fg'].get('alpha_sync_BB', -0.4)
    beta_sync = info['fg'].get('beta_sync', -3.1)
    A_sync_EE = EB_sync * A_sync_BB
    A_dust_BB = info['fg'].get('A_dust_BB', 5.0)
    EB_dust = info['fg'].get('EB_dust', 2.0)
    alpha_dust_EE = info['fg'].get('alpha_dust_EE', -0.42)
    alpha_dust_BB = info['fg'].get('alpha_dust_BB', -0.2)
    A_dust_EE = EB_dust * A_dust_BB
    Alens = info['fg'].get('Alens', 1.)
    r = info['fg'].get('r', 0.)
    beta_sync = info['fg'].get('beta_sync', -3.1)
    nu0_sync = 23.
    beta_dust = info['fg'].get('beta_dust', 1.59)
    temp_dust = info['fg'].get('temp_dust', 19.6)
    nu0_dust = 353.
    use_moments = info['fg'].get("use_moments", False)
    sampler = info.get('sampler', 'single_point')
    nwalkers = info.get('nwalkers', 128)
    nsamples = info.get('nsamples', 10000)

    stout=""
    stout+="# These parameters are accessible to all stages\n"
    stout+="global:\n"
    stout+="    # HEALPix resolution parameter\n"
    stout+="    nside: 512\n"
    stout+="    # Use D_l = l*(l+1)*C_l/(2*pi) instead of C_l?\n"
    stout+="    compute_dell: True\n"
    stout+="\n"
    stout+="BBCompSep:\n"
    stout+="    # Sampler type. Options are:\n"
    stout+="    #  - 'emcee': a full MCMC will be run using emcee.\n"
    stout+="    #  - 'fisher': only the Fisher matrix (i.e. the likelihood\n"
    stout+="    #        Hessian matrix) will be calculated around the fiducial\n"
    stout+="    #        parameters chosen as prior centers.\n"
    stout+="    #  - 'maximum_likelihood': only the best-fit parameters will\n"
    stout+="    #        be searched for.\n"
    stout+="    #  - 'single_point': only the chi^2 at the value used as the\n"
    stout+="    #        center for all parameter priors will be calculated.\n"
    stout+="    #  - 'timing': will compute the average time taken by one\n"
    stout+="    #        likelihood computation.\n"
    stout+=f"    sampler: {sampler}\n"
    stout+="    # If you chose emcee:\n"
    stout+="    # Number of walkers\n"
    stout+=f"    nwalkers: {nwalkers}\n"
    stout+="    # Number of iterations per walker\n"
    stout+=f"    n_iters: {nsamples}\n"
    stout+="    # Likelihood type. Options are:\n"
    stout+="    #  - 'chi2': a standard chi-squared Gaussian likelihood.\n"
    stout+="    #  - 'h&l': Hamimeche & Lewis likelihood.\n"
    stout+="    likelihood_type: 'chi2'\n"
    stout+="    # Which polarization channels do you want to include?\n"
    stout+="    # Can be ['E'], ['B'] or ['E','B'].\n"
    stout+="    pol_channels: ['B']\n"
    stout+="    # Scale cuts (will apply to all frequencies)\n"
    stout+="    l_min: 30\n"
    stout+="    l_max: 300\n"
    stout+="\n"
    stout+="    # CMB model\n"
    stout+="    cmb_model:\n"
    stout+="        # Template power spectrum. Should contained the lensed power spectra\n"
    stout+="        # with r=0 and r=1 respectively.\n"
    stout+="        cmb_templates: [\"./examples/data/camb_lens_nobb.dat\", \n"
    stout+="                        \"./examples/data/camb_lens_r1.dat\"]\n"
    stout+="        # Free parameters\n"
    stout+="        params:\n"
    stout+="            # tensor-to-scalar ratio\n"
    stout+="            # See below for the meaning of the different elements in the list.\n"
    stout+=f"            r_tensor: ['r_tensor', 'tophat', [-0.1, {r}, 0.1]]\n"
    stout+="            # Lensing amplitude\n"
    stout+=f"            A_lens: ['A_lens', 'tophat', [0.00,{Alens},2.00]]\n"
    stout+="\n"
    stout+="    # Foreground model\n"
    stout+="    fg_model:\n"
    stout+="        # Include moment parameters?\n"
    stout+=f"        use_moments: {use_moments}\n"
    stout+="        moments_lmax: 384\n"
    stout+="\n"
    stout+="        # Add one section per component. They should be called `component_X`,\n"
    stout+="        # starting with X=1\n"
    stout+="        component_1:\n"
    stout+="            # Name for this component\n"
    stout+="            name: Dust\n"
    stout+="            # Type of SED. Should be one of the classes stored in fgbuster.components\n"
    stout+="            # https://github.com/fgbuster/fgbuster/blob/master/fgbuster/component_model.py\n"
    stout+="            sed: Dust\n"
    stout+="            # Type of power spectra for all possible polarization channel combinations.\n"
    stout+="            # Any combinations not added here will be assumed to be zero.\n"
    stout+="            # The names should be one of the classes in bbpower/fgcls.py. This is quite\n"
    stout+="            # limiter for now, so consider adding to it if you want something fancier.\n"
    stout+="            cl:\n"
    stout+="                EE: ClPowerLaw\n"
    stout+="                BB: ClPowerLaw\n"
    stout+="            # Parameters of the SED\n"
    stout+="            sed_parameters:\n"
    stout+="                # The key can be anything you want, but each parameter in the model\n"
    stout+="                # must have a different name.\n"
    stout+="                # The first item in the list is the name of the parameter used by fgbuster\n"
    stout+="                # The second item is the type of prior. The last item are the numbers\n"
    stout+="                # necessary to define the prior. They should be:\n"
    stout+="                #  - Gaussian: [mean,sigma]\n"
    stout+="                #  - tophat: [lower edge, start value, upper edge]\n"
    stout+="                #  - fixed: [parameter value]\n"
    stout+="                # nu0-type parameters can only be fixed.\n"
    stout+=f"                beta_d: ['beta_d', 'Gaussian', [{beta_dust}, 0.11]]\n"
    stout+=f"                temp_d: ['temp', 'fixed', [{temp_dust}]]\n"
    stout+=f"                nu0_d: ['nu0', 'fixed', [{nu0_dust}]]\n"
    stout+="            cl_parameters:\n"
    stout+="                # Same for power spectrum parameters\n"
    stout+="                # (broken down by polarization channel combinations)\n"
    stout+="                EE:\n"
    stout+=f"                   amp_d_ee: ['amp', 'tophat', [0., {A_dust_EE}, \"inf\"]]\n"
    stout+=f"                   alpha_d_ee: ['alpha', 'tophat', [-1., {alpha_dust_EE}, 0.]]\n"
    stout+="                   l0_d_ee: ['ell0', 'fixed', [80.]]\n"
    stout+="                BB:\n"
    stout+=f"                   amp_d_bb: ['amp', 'tophat', [0., {A_dust_BB}, \"inf\"]]\n"
    stout+=f"                   alpha_d_bb: ['alpha', 'tophat', [-1., {alpha_dust_BB}, 0.]]\n"
    stout+="                   l0_d_bb: ['ell0', 'fixed', [80.]]\n"
    stout+="            # If this component should be correlated with any other, list them here\n"
    stout+="            cross:\n"
    stout+="                # In this case the list should contain:\n"
    stout+="                # [component name, prior type, prior parameters]\n"
    stout+="                # Each of this will create a new parameter, corresponding to a constant\n"
    stout+="                # scale- and frequency-independend correlation coefficient between\n"
    stout+="                # the two components.\n"
    stout+="                epsilon_ds: ['component_2', 'tophat', [-1., 0., 1.]]\n"
    stout+="            moments:\n"
    stout+="                # Define gammas for varying spectral indices of components\n"
    stout+="                gamma_d_beta : ['gamma_beta', 'tophat', [-6., -3.5, -2.]]\n"
    stout+="                amp_d_beta : ['amp_beta', 'tophat', [0., 0., 1]]\n"
    stout+="\n"
    stout+="        component_2:\n"
    stout+="            name: Synchrotron\n"
    stout+="            sed: Synchrotron\n"
    stout+="            cl:\n"
    stout+="                EE: ClPowerLaw\n"
    stout+="                BB: ClPowerLaw\n"
    stout+="            sed_parameters:\n"
    stout+=f"                beta_s: ['beta_pl', 'Gaussian', [{beta_sync}, 0.3]]\n"
    stout+=f"                nu0_s: ['nu0', 'fixed', [{nu0_sync}]]\n"
    stout+="            cl_parameters:\n"
    stout+="                EE:\n"
    stout+=f"                    amp_s_ee: ['amp', 'tophat', [0., {A_sync_EE}, \"inf\"]]\n"
    stout+=f"                    alpha_s_ee: ['alpha', 'tophat', [-1., {alpha_sync_EE}, 0.]]\n"
    stout+="                    l0_s_ee: ['ell0', 'fixed', [80.]]\n"
    stout+="                BB:\n"
    stout+=f"                    amp_s_bb: ['amp', 'tophat', [0., {A_sync_BB}, \"inf\"]]\n"
    stout+=f"                    alpha_s_bb: ['alpha', 'tophat', [-1., {alpha_sync_BB}, 0.]]\n"
    stout+="                    l0_s_bb: ['ell0', 'fixed', [80.]]\n"
    stout+="            moments:\n"
    stout+="                # Define gammas for varying spectral indices of components\n"
    stout+="                gamma_s_beta : ['gamma_beta', 'tophat', [-6., -3.5, -2.]]\n"
    stout+="                amp_s_beta : ['amp_beta', 'tophat', [0., 0., 1]]\n"
    stout+="\n"
    f = open(info['prefix_out'] + '/config.yml', 'w')
    f.write(stout)


params = {}
for fname in sys.argv[1:-1]:
    with open(fname) as f:
        d = yaml.safe_load(f)
        params.update(d)
params.update({'prefix_out': sys.argv[-1]})

os.system('mkdir -p ' + params['prefix_out'])
print("Generating data")
get_spectra(params)
print("Setting up BBPower")
get_bbpower_config(params)
cmd = get_bbpower_command(params)
print("Running BBPower")
os.system(cmd)

import numpy as np
import matplotlib.pyplot as plt
from V3calc import so_V3_SA_noise, so_V3_SA_bands, so_V3_SA_beams
import sacc
import healpy as hp
import pymaster as nmt
import os


def get_pairs():
    ix = 0
    for i1 in range(6):
        for i2 in range(i1, 6):
            yield i1, i2, ix
            ix += 1


def get_bandpower_windows(nls, delta_ell=10, is_dl=False):
    """ Returns a binned-version of the power spectra.
    """
    N_bins = (nls-2)//delta_ell
    if is_dl:
        ls = np.arange(nls)
        w = ls*(ls+1)/(2*np.pi*delta_ell)
    else:
        w = np.ones(nls)/delta_ell
    W = np.zeros([N_bins, nls])
    for i in range(N_bins):
        w_h = w[2+i*delta_ell:2+(i+1)*delta_ell]
        W[i, 2+i*delta_ell:2+(i+1)*delta_ell] = w_h
    return W


def get_beams(config, inv=False):
    nside = config['nside']
    ls = np.arange(3*nside)
    b_fwhm_amin = so_V3_SA_beams()
    b_sigma_rad = np.radians(b_fwhm_amin/(2.355*60))
    exponent = 0.5*(ls*(ls+1))[None, :]*(b_sigma_rad**2)[:, None]
    if inv:
        return np.exp(exponent)
    else:
        return np.exp(-exponent)


def get_noise(config, fsky, beamed=False):
    noi_codes = {'baseline': 1,
                 'goal': 2}
    ooe_codes = {'pessimistic': 0,
                 'optimistic': 1}
    l, nl, _ = so_V3_SA_noise(noi_codes[config['noise_level']],
                              ooe_codes[config['ell_knee']],
                              config.get('SAT_years', 1),
                              fsky, 3*config['nside'],
                              beam_corrected=beamed,
                              remove_kluge=True)
    nls = np.zeros([6, 3*config['nside']])
    nls[:, l] = nl
    return nls


def fcmb(nu):
    """ CMB SED (in antenna temperature units).
    """
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2


def comp_sed(nu,nu0,beta,temp,typ):
    """ Component SEDs (in antenna temperature units).
    """
    if typ=='cmb':
        return fcmb(nu)
    elif typ=='dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)
    elif typ=='sync':
        return (nu/nu0)**beta
    return None


def get_cls(config):
    # Global
    nside = config['nside']
    ls = np.arange(3*nside)
    l0 = config.get('l0', 80.)
    dl2cl = np.ones(len(ls))
    dl2cl[1:] = 2*np.pi/(ls[1:]*(ls[1:]+1.))
    cl2dl = (ls*(ls+1.))/(2*np.pi)
    nu = so_V3_SA_bands()

    # CMB
    Alens = config.get('Alens', 1.)
    l,dtt,dee,dbb,dte=np.loadtxt("data/camb_lens_nobb.dat",unpack=True)
    l = l.astype(int)
    msk = l < 3*nside
    l = l[msk]
    dl_c_BB=np.zeros(len(ls))
    dl_c_BB[l]=Alens*dbb[msk]
    dl_c_EE=np.zeros(len(ls))
    dl_c_EE[l]=dee[msk]
    cl_c_BB = dl_c_BB*dl2cl
    cl_c_EE = dl_c_EE*dl2cl
    f_c = fcmb(nu)

    # Dust
    beta_d = config['beta_d']
    temp_d = config['temp_d']
    A_d_B = config['A_d_B']
    A_d_E = config.get('A_d_E', 2*A_d_B)
    alpha_d = config['alpha_d']
    nu0_d = config.get('nu0_d', 353.)
    A_d_B *= fcmb(nu0_d)**2
    A_d_E *= fcmb(nu0_d)**2
    dl_d_BB = A_d_B * ((ls+0.1)/l0)**alpha_d
    dl_d_EE = A_d_E * ((ls+0.1)/l0)**alpha_d
    cl_d_BB = dl_d_BB*dl2cl
    cl_d_EE = dl_d_EE*dl2cl
    f_d = comp_sed(nu, nu0_d, beta_d, temp_d, 'dust')

    # Synchrotron
    beta_s = config['beta_s']
    A_s_B = config['A_s_B']
    A_s_E = config.get('A_s_E', 2*A_s_B)
    alpha_s = config['alpha_s']
    nu0_s = config.get('nu0_s', 23.)
    A_s_B *= fcmb(nu0_s)**2
    A_s_E *= fcmb(nu0_s)**2
    dl_s_BB = A_s_B * ((ls+0.1)/l0)**alpha_s
    dl_s_EE = A_s_E * ((ls+0.1)/l0)**alpha_s
    cl_s_BB = dl_s_BB*dl2cl
    cl_s_EE = dl_s_EE*dl2cl
    f_s = comp_sed(nu, nu0_s, beta_s, None, 'sync')

    return {'l': ls, 'dl2cl': dl2cl, 'cl2dl': cl2dl,
            'CMB': {'EE': cl_c_EE, 'BB': cl_c_BB,
                    'sed': f_c, 'sed_n': f_c/f_c},
            'Dust': {'EE': cl_d_EE, 'BB': cl_d_BB,
                    'sed': f_d, 'sed_n': f_d/f_c},
            'Sync': {'EE': cl_s_EE, 'BB': cl_s_BB,
                     'sed': f_s, 'sed_n': f_s/f_c}}


def get_sacc_theory(config, fsky):
    nside = config['nside']
    ls = np.arange(3*nside)
    nls = get_noise(config, fsky, beamed=True)
    sky = get_cls(config)
    delta_ell = 10
    W = get_bandpower_windows(3*nside,
                              delta_ell=delta_ell,
                              is_dl=True).T
    l_bpw = np.dot(get_bandpower_windows(3*nside,
                                         delta_ell=delta_ell),
                   ls)
    sl_matrix = np.zeros([6, 6, 3*nside])
    for comp in ['CMB', 'Dust', 'Sync']:
        d = sky[comp]
        sl_matrix += d['BB'][None, None, :]*d['sed_n'][None, :, None]*d['sed_n'][:, None, None]
    nl_matrix = np.zeros([6, 6, 3*nside])
    for i in range(6):
        nl_matrix[i, i, :] = nls[i]
    bpw_s = np.dot(sl_matrix, W)
    bpw_c = np.dot(sl_matrix+nl_matrix, W)

    beams = get_beams(config)
    nus = so_V3_SA_bands()
    s = sacc.Sacc()
    bpwf = sacc.BandpowerWindow(ls, W)
    for i, nu in enumerate(nus):
        s.add_tracer('NuMap', 'band%d' % i,
                     quantity='cmb_polarization',
                     spin=2,
                     nu=np.array([nu-1, nu, nu+1]),
                     bandpass=np.array([0., 1., 0.]),
                     ell=ls, beam=beams[i],
                     nu_unit='GHz', map_unit='uK_CMB')

    for i1, i2, ix in get_pairs():
        s.add_ell_cl('cl_bb', 'band%d'%i1, 'band%d'%i2, l_bpw, bpw_s[i1, i2], window=bpwf)

    nx = 6*7//2
    nbpw = len(l_bpw)
    cov = np.zeros([nx, nbpw, nx, nbpw])
    nmodes = (2*l_bpw+1)*delta_ell*fsky
    for i1, i2, ix in get_pairs():
        for j1, j2, jx in get_pairs():
            ci1j1 = bpw_c[i1, j1]
            ci1j2 = bpw_c[i1, j2]
            ci2j1 = bpw_c[i2, j1]
            ci2j2 = bpw_c[i2, j2]
            cov[ix, :, jx, :] = np.diag((ci1j1*ci2j2+ci1j2*ci2j1)/nmodes)
    cov = cov.reshape([nx*nbpw, nx*nbpw])
    s.add_covariance(cov)
    return s


def get_circular_mask(config, th0=np.pi/2, phi0=0.):
    nside = config['nside']
    fsky = config['fsky']
    aposize = config['aposize']
    npix = hp.nside2npix(nside)
    alpha = np.arccos(1-2*fsky*1.28)  # This makes it an even 0.1
    v0 = np.array(hp.ang2vec(th0, phi0))
    v = np.array(hp.pix2vec(nside, np.arange(npix)))
    x = (alpha-np.arccos(np.dot(v0, v)))/np.radians(aposize)
    mask = x-np.sin(2*np.pi*x)/(2*np.pi)
    mask[x < 0] = 0
    mask[x > 1] = 1
    return mask
    

def get_simulation(config, seed, recompute=False, get_bpwws=False):
    fname_out = config['output_dir'] + f'cls_sim{seed}.npz'
    if os.path.isfile(fname_out) and (not recompute):
        print(f" Reading simulation {seed} from file")
        d = np.load(fname_out)
        if get_bpwws:
            bpwws = np.array([np.load(config['output_dir'] + f'bpwws{i1}{i2}.npz')['bpww']
                              for i1, i2, ix in get_pairs()])
        else:
            bpwws = None
        return d['ls'], d['cls_s'], d['cls_n'], bpwws

    print(f" Generating simulation {seed}")
    np.random.seed(seed)

    nside = config['nside']
    npix = hp.nside2npix(nside)
    if config['mask_type'] == 'circular' :
        mask = get_circular_mask(config)
    elif config['mask_type'] == 'SAT':
        mask = hp.ud_grade(hp.read_map("data/mask_apodized.fits", verbose=False),
                           nside_out=nside)
    else:
        raise NotImplementedError("Not yet")

    use_nhits = config.get('use_nhits', False)
    if use_nhits:
        nhits = hp.ud_grade(hp.read_map("data/norm_nHits_SA_35FOV.fits", verbose=False),
                            nside_out=nside)
        fsky_use = np.mean(nhits)
    else:
        fsky_use = config['fsky']
    nls = get_noise(config, fsky_use)
    sky = get_cls(config)

    # Create noise maps
    if config.get('scalar_B', True):
        noise_maps = np.array([hp.synfast(n, nside, new=True, verbose=False)
                               for n in nls])
        if use_nhits:
            id_good = nhits > 0.001 * np.amax(nhits)
            noise_maps[:, id_good] = noise_maps[:, id_good] / np.sqrt(nhits[id_good])
    else:
        noise_maps = np.array([[hp.synfast(n, nside, new=True, verbose=False),
                                hp.synfast(n, nside, new=True, verbose=False)]
                               for n in nls])
        if use_nhits:
            id_good = nhits > 0.001 * np.amax(nhits)
            noise_maps[:, :, id_good] = noise_maps[:, :, id_good] / np.sqrt(nhits[id_good])

    # Create component alms
    comp_alms = {}
    if config.get('scalar_B', True):
        for comp in ['CMB', 'Dust', 'Sync']:
            comp_alms[comp] = hp.synalm(sky[comp]['BB'], new=True)
    else:
        for comp in ['CMB', 'Dust', 'Sync']:
            comp_alms[comp] = np.array([hp.synalm(sky[comp]['EE'], new=True),
                                        hp.synalm(sky[comp]['BB'], new=True)])

    # Create frequency signal maps
    beams = get_beams(config)
    if config.get('scalar_B', True):
        signal_maps = np.zeros([6, npix])
        for inu, nu in enumerate(so_V3_SA_bands()):
            beam = beams[inu]
            # Convolve with beam and transform
            comp_maps = {k: hp.alm2map(hp.almxfl(alm, beam), nside, verbose=False)
                         for k, alm in comp_alms.items()}
            for k, m in comp_maps.items():
                sed = sky[k]['sed_n'][inu]
                signal_maps[inu] += m*sed
    else:
        signal_maps = np.zeros([6, 2, npix])
        for inu, nu in enumerate(so_V3_SA_bands()):
            beam = beams[inu]
            # Convolve with beam and transform
            comp_maps = {k: np.array(hp.alm2map_spin([hp.almxfl(alm, beam) for alm in alms],
                                                     nside, spin=2, lmax=3*nside-1))
                         for k, alms in comp_alms.items()}
            for k, m in comp_maps.items():
                sed = sky[k]['sed_n'][inu]
                signal_maps[inu, :, :] += m*sed

    # Mix it up
    ZERO = 0
    mask_binary = np.ones_like(mask)
    mask_binary[mask <= ZERO] = 0
    data_maps = mask_binary*(signal_maps+noise_maps)

    # Power spectra
    # Bandpowers
    bins = nmt.NmtBin(nside, nlb=10, is_Dell=True)
    # Fields
    if config.get('scalar_B', True): 
        fs = [nmt.NmtField(mask, [m], beam=b, n_iter=config.get('n_iter', 0))
              for m, b in zip(data_maps, beams)]
        fn = [nmt.NmtField(mask, [m], beam=b, n_iter=config.get('n_iter', 0))
              for m, b in zip(noise_maps, beams)]        
    else:
        fs = [nmt.NmtField(mask, m, beam=b, n_iter=config.get('n_iter', 0), purify_b=True)
              for m, b in zip(data_maps, beams)]
        fn = [nmt.NmtField(mask, m, beam=b, n_iter=config.get('n_iter', 0), purify_b=True)
              for m, b in zip(noise_maps, beams)]

    # Workspaces and c_ells
    cls_s = []
    cls_n = []
    if get_bpwws:
        bpwws = []
    else:
        bpwws = None
    for i1, i2, ix in get_pairs():
        fs1 = fs[i1]
        fs2 = fs[i2]
        fn1 = fn[i1]
        fn2 = fn[i2]
        w = nmt.NmtWorkspace()
        fname_w = config['output_dir'] + f'wsp{i1}{i2}.fits'
        if os.path.isfile(fname_w):
            w.read_from(fname_w)
        else:
            w.compute_coupling_matrix(fs1, fs2, bins)
            w.write_to(fname_w)
        if get_bpwws:
            fname_b = config['output_dir'] + f'bpwws{i1}{i2}.npz'
            if os.path.isfile(fname_b):
                d = np.load(fname_b)
                bpww = d['bpww']
            else:
                bpww = w.get_bandpower_windows()
                np.savez(fname_b, bpww=bpww)
            bpwws.append(bpww)
        cls = w.decouple_cell(nmt.compute_coupled_cell(fs1, fs2))
        cln = w.decouple_cell(nmt.compute_coupled_cell(fn1, fn2))
        cls_s.append(cls)
        cls_n.append(cln)
    cls_s = np.array(cls_s)
    cls_n = np.array(cls_n)
    if get_bpwws:
        bpwws = np.array(bpwws)
    ls = bins.get_effective_ells()

    np.savez(fname_out, ls=ls, cls_s=cls_s, cls_n=cls_n)

    return ls, cls_s, cls_n, bpwws


config = {'noise_level': 'baseline',
          'ell_knee': 'optimistic',
          'nside': 256,
          'A_d_B': 28., 'alpha_d': -0.16, 'beta_d': 1.6, 'temp_d': 19.6,
          'A_s_B': 1.6, 'alpha_s': -0.93, 'beta_s': -3.,
          'mask_type': 'SAT', 'fsky': 0.1, 'aposize': 10, 'use_nhits': True,
          'scalar_B': False, 'nsims': 500, 'sims_to_save': [0, 1, 2, 3, 4, 5],
          'output_dir': 'sim_SAT_B2_inhomogeneous/'}
#config = {'noise_level': 'baseline',
#          'ell_knee': 'optimistic',
#          'nside': 256,
#          'A_d_B': 28., 'alpha_d': -0.16, 'beta_d': 1.6, 'temp_d': 19.6,
#          'A_s_B': 1.6, 'alpha_s': -0.93, 'beta_s': -3.,
#          'mask_type': 'SAT', 'fsky': 0.1, 'aposize': 10, 'use_nhits': True,
#          'scalar_B': True, 'nsims': 500, 'sims_to_save': [0, 1, 2, 3, 4, 5],
#          'output_dir': 'sim_SAT_B0_inhomogeneous/'}
#config = {'noise_level': 'baseline',
#          'ell_knee': 'optimistic',
#          'nside': 256,
#          'A_d_B': 28., 'alpha_d': -0.16, 'beta_d': 1.6, 'temp_d': 19.6,
#          'A_s_B': 1.6, 'alpha_s': -0.93, 'beta_s': -3.,
#          'mask_type': 'SAT', 'fsky': 0.1, 'aposize': 10,
#          'scalar_B': True, 'nsims': 500, 'sims_to_save': [0, 1, 2, 3, 4, 5],
#          'output_dir': 'sim_SAT_B0_homogeneous/'}
#config = {'noise_level': 'baseline',
#          'ell_knee': 'optimistic',
#          'nside': 256,
#          'A_d_B': 28., 'alpha_d': -0.16, 'beta_d': 1.6, 'temp_d': 19.6,
#          'A_s_B': 1.6, 'alpha_s': -0.93, 'beta_s': -3.,
#          'mask_type': 'circular', 'fsky': 0.1, 'aposize': 10,
#          'scalar_B': True, 'nsims': 500, 'sims_to_save': [0, 1, 2, 3, 4, 5],
#          'output_dir': 'sim_circular_B0_homogeneous/'}

os.system('mkdir -p ' + config['output_dir'])
# Get theory
print("Theory power spectra")
sth = get_sacc_theory(config, 0.1)
sth.save_fits(config['output_dir'] + 'theory_sacc.fits', overwrite=True)

# Generate simulations
cls_s = []
cls_n = []
bpwws = None
for i in range(config['nsims']):
    get_bpwws = i == 0
    l_bpw, cls, cln, w = get_simulation(config, 1000+i, recompute=False, get_bpwws=get_bpwws)
    if get_bpwws:
        bpwws = w
    cls_s.append(cls)
    cls_n.append(cln)
cls_s = np.array(cls_s)
cls_n = np.array(cls_n)
cl_n = np.mean(cls_n, axis=0)
cl_s = np.mean(cls_s, axis=0)

# Subtract mean noise and compute covariance
if config.get('scalar_B', True):
    icl = 0
else:
    icl = 3
nsim, ncls, nmod, nls = cls_s.shape
cls_s -= cl_n[None, :, :, :]
cl_s -= cl_n
covar_all = np.cov(cls_s.reshape([nsim, ncls*nmod*nls]).T)
covar_all = covar_all.reshape([ncls, nmod, nls, ncls, nmod, nls])
covar = np.zeros([ncls, nls, ncls, nls])
# Keep diagonal only
for i1, i2, ix in get_pairs():
    for j1, j2, jx in get_pairs():
        covar[ix, :, jx, :] = np.diag(np.diag(covar_all[ix, icl, :, jx, icl, :]))
covar = covar.reshape([ncls*nls, ncls*nls])

# Save sacc files
ls = np.arange(3*config['nside'])
saccs = []
W = get_bandpower_windows(3*config['nside'],
                          delta_ell=10,
                          is_dl=True).T
#wins = [sacc.BandpowerWindow(ls, W) for i1, i2, ix in get_pairs()]
wins = [sacc.BandpowerWindow(ls, bpwws[ix][icl, :, icl, :].T)
        for i1, i2, ix in get_pairs()]
print(bpwws[0].shape)
for isim in config['sims_to_save']:
    print(isim)
    s = sacc.Sacc()
    nus = so_V3_SA_bands()
    beams = get_beams(config)
    for i, nu in enumerate(nus):
        s.add_tracer('NuMap', 'band%d' % i,
                     quantity='cmb_polarization',
                     spin=2,
                     nu=np.array([nu-1, nu, nu+1]),
                     bandpass=np.array([0., 1., 0.]),
                     ell=ls, beam=beams[i],
                     nu_unit='GHz', map_unit='uK_CMB')

    for i1, i2, ix in get_pairs():
        s.add_ell_cl('cl_bb', 'band%d'%i1, 'band%d'%i2, l_bpw, cls_s[isim, ix, icl, :], window=wins[ix])
    s.add_covariance(covar)
    seed = 1000 + isim
    s.save_fits(config['output_dir'] + f'sim{seed}_sacc.fits', overwrite=True)
    saccs.append(s)

for i1, i2, ix in get_pairs():
    lth, clth, cvth = sth.get_ell_cl('cl_bb', 'band%d' % i1, 'band%d' % i2, return_cov=True)
    plt.figure()
    for s  in saccs:
        l, cl, cv = s.get_ell_cl('cl_bb', 'band%d' % i1, 'band%d' % i2, return_cov=True)
        #plt.plot(l, cl)
        plt.plot(l, np.diag(cv), 'r-')
    #plt.plot(lth, clth, 'k-', lw=2)
    #plt.plot(lth, cl_s[ix, icl], 'r-', lw=2)
    plt.plot(l, np.diag(cvth), 'k-')
    plt.loglog()
plt.show()

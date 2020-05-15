import numpy as np
from utils import *
import noise_calc as nc
import sacc
import sys


prefix_out = sys.argv[1]

# Bandpasses
bpss = {n: Bpass(n,f'examples/data/bandpasses/{n}.txt') for n in band_names}

# Bandpowers
dell = 10
nbands = 100
lmax = 2+nbands*dell
larr_all = np.arange(lmax+1)
lbands = np.linspace(2,lmax,nbands+1,dtype=int)
leff = 0.5*(lbands[1:]+lbands[:-1])
windows = np.zeros([nbands,lmax+1])
cl2dl=larr_all*(larr_all+1)/(2*np.pi)
dl2cl=np.zeros_like(cl2dl)
dl2cl[1:] = 1/(cl2dl[1:])
for b,(l0,lf) in enumerate(zip(lbands[:-1],lbands[1:])):
    windows[b,l0:lf] = (larr_all * (larr_all + 1)/(2*np.pi))[l0:lf]
    windows[b,:] /= dell
s_wins = sacc.BandpowerWindow(larr_all, windows.T)

# Beams
beams = {band_names[i]: b for i, b in enumerate(nc.Simons_Observatory_V3_SA_beams(larr_all))}

print("Calculating power spectra")
# Component spectra
dls_comp=np.zeros([3,2,3,2,lmax+1]) #[ncomp,np,ncomp,np,nl]
(dls_comp[1,0,1,0,:],
 dls_comp[1,1,1,1,:],
 dls_comp[2,0,2,0,:],
 dls_comp[2,1,2,1,:],
 dls_comp[0,0,0,0,:],
 dls_comp[0,1,0,1,:]) = get_component_spectra(lmax)
dls_comp *= dl2cl[None, None, None, None, :]

# Convolve with windows
bpw_comp=np.sum(dls_comp[:,:,:,:,None,:]*windows[None,None,None,None,:,:],axis=5)

# Convolve with bandpasses
seds = get_convolved_seds(band_names, bpss)
_, nfreqs = seds.shape

# Component -> frequencies
bpw_freq_sig=np.einsum('ik,jm,iljno',seds,seds,bpw_comp)

# N_ell
sens=1
knee=1
ylf=1
fsky=0.1
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=nc.Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1)
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
print("Adding tracers")
for ib, n in enumerate(band_names):
    bandpass = bpss[n]
    beam = beams[n]
    for s in [s_d, s_f, s_n]:
        s.add_tracer('NuMap', 'band%d' % (ib+1),
                     quantity='cmb_polarization',
                     spin=2,
                     nu=bandpass.nu,
                     bandpass=bandpass.bnu,
                     ell=larr_all,
                     beam=beam,
                     nu_unit='GHz',
                     map_unit='uK_CMB')

# Adding power spectra
print("Adding spectra")
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
print("Adding covariance")
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
print("Writing")
s_d.save_fits(prefix_out + "/cls_coadd.fits", overwrite=True)
s_f.save_fits(prefix_out + "/cls_fid.fits", overwrite=True)
s_n.save_fits(prefix_out + "/cls_noise.fits", overwrite=True)

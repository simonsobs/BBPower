from utils import *
import numpy as np
import noise_calc as nc
from optparse import OptionParser
import healpy as hp

parser = OptionParser()
parser.add_option('--output-dir', dest='dirname', default='none',
                  type=str, help='Output directory')
parser.add_option('--seed', dest='seed',  default=1000, type=int,
                  help='Set to define seed, default=1000')
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Set to define Nside parameter, default=256')
(o, args) = parser.parse_args()

np.random.seed(o.seed)
npix = hp.nside2npix(o.nside)

# Signal maps
lmax = 3*o.nside - 1
larr = np.arange(lmax+1)
cl2dl=larr*(larr+1)/(2*np.pi)
dl2cl = np.zeros(lmax+1)
dl2cl[1:]=2*np.pi/(larr[1:]*(larr[1:]+1))
clsee, clsbb, cldee, cldbb, clcee, clcbb = get_component_spectra(lmax)
clsee *= dl2cl
clsbb *= dl2cl
cldee *= dl2cl
cldbb *= dl2cl
clcee *= dl2cl
clcbb *= dl2cl
cl0 = 0*clsee
_, Qs, Us = hp.synfast([cl0, clsee, clsbb, cl0, cl0, cl0],
                       o.nside, verbose=False, new=True)
_, Qd, Ud = hp.synfast([cl0, cldee, cldbb, cl0, cl0, cl0],
                       o.nside, verbose=False, new=True)
_, Qc, Uc = hp.synfast([cl0, clcee, clcbb, cl0, cl0, cl0],
                       o.nside, verbose=False, new=True)
map_comp = np.array([[Qc, Uc],
                     [Qs, Us],
                     [Qd, Ud]])
bpss = {n: Bpass(n, f'examples/data/bandpasses/{n}.txt')
        for n in band_names}
seds = get_convolved_seds(band_names, bpss)
_, nfreqs = seds.shape
map_freq = np.einsum('ij,ikl', seds, map_comp)

# Noise maps
nsplits = 4
sens=1
knee=1
ylf=1
fsky=0.1
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=nc.Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1,
                                                 include_beam=False)
map_noise = np.zeros([nsplits, nfreqs, 2, npix])
for i_s in range(nsplits):
    for i_f in range(nfreqs):
        _, mpq, mpu = hp.synfast([cl0, nell[i_f], nell[i_f], cl0, cl0, cl0],
                                 o.nside, verbose=False, new=True)
        map_noise[i_s, i_f, 0, :] = mpq * np.sqrt(nsplits)
        map_noise[i_s, i_f, 1, :] = mpu * np.sqrt(nsplits)

# Beam convolution
s_fwhm = nc.Simons_Observatory_V3_SA_beam_FWHM()
for i_f, s in enumerate(s_fwhm):
    fwhm = s * np.pi/180./60.
    for i_p in [0, 1]:
        map_freq[i_f, i_p, :] = hp.smoothing(map_freq[i_f, i_p, :],
                                             fwhm=fwhm, verbose=False)

# Write output
for s in range(nsplits):
    m = (map_freq + map_noise[s]).reshape([nfreqs*2, npix])
    hp.write_map(o.dirname+'/obs_split%dof%d.fits.gz' % (s+1, nsplits),
                 m, overwrite=True, dtype=np.float32)

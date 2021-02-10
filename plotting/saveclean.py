import os.path
import numpy as np
import glob
import emcee
from labels import alllabels

#allfs = glob.glob('/mnt/zfsusers/mabitbol/BBPower/residual_noiseless/sim0*/output*/')
#allfs.sort()

allfs = glob.glob('/mnt/zfsusers/mabitbol/BBPower/baseline_noiseless/sim0*/output*/')
allfs.sort()

dones = []
for af in allfs:
    if os.path.isfile(af+'cleaned_chains.npz'):
        dones.append(af)
for df in dones:
    allfs.remove(df)
print(len(allfs))


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    if norm:
        acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def save_cleaned_chains(fdir):
    outf = fdir+'chain_info.txt'
    reader = emcee.backends.HDFBackend(fdir+'param_chains.npz.h5')
    x = np.load(fdir+'param_chains.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]

    tau0 = reader.get_autocorr_time(tol=0)
    burnin0 = int(5. * np.max(tau0))
    samps = reader.get_chain(discard=burnin0)

    nm = samps.shape[0]
    nc = samps.shape[1]
    npar = samps.shape[2]

    tau = np.zeros(npar)
    for k in range(npar):
        tau[k] = autocorr_new(samps[:, :, k].T)

    burnin = int(5. * np.max(tau))
    thin = int(0.5 * np.max(tau))

    samples = reader.get_chain(discard=burnin, flat=True, thin=thin) 
    N = samples.shape[0]

    with open(outf, 'w') as of:
        inds = nm/tau
        print("N: ", N, file=of)
        print("burnin: %d, thin: %d" %(burnin, thin), file=of)
        print("mean tau: ", np.mean(tau), file=of)
        print("mean number independent samps per chain: %d" %np.mean(inds), file=of)
        print("percent Mean chain uncertainty: ", 100./np.sqrt(N), file=of)
        print("\n", file=of)
        print("number independent samps per chain:", file=of)
        print(inds, file=of)
        print("tau:", file=of)
        print(tau, file=of)
        print("percent parameter uncertainty: ", file=of)
        print(100./np.sqrt(inds), file=of)
        if np.any(inds < 50): 
            print("POTENTIALLY BAD TAU", file=of)
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels)
    return

for af in allfs:
    save_cleaned_chains(af)



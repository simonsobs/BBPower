import numpy as np
import emcee
from labels import alllabels
import glob

allfs = ['/mnt/zfsusers/mabitbol/BBPower/moments/hybrid0chi2p3/output/']


def save_cleaned_chains(fdir):
    outf = fdir+'chain_info.txt'
    reader = emcee.backends.HDFBackend(fdir+'param_chains.npz.h5')
    x = np.load(fdir+'param_chains.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]

    tau = reader.get_autocorr_time(tol=0)
    burnin = int(10. * np.mean(tau))
    thin = int(0.5 * np.mean(tau))

    samples = reader.get_chain(discard=burnin, flat=True, thin=thin) 
    chains = reader.get_chain(discard=burnin, flat=False) 

    N = chains.shape[0]
    M = chains.shape[1]
    chain_mean = np.mean(chains, axis=0)
    chain_var = np.var(chains, axis=0)
    samp_mean = np.mean(chains, axis=(0,1))
    B = N / (M-1) * np.sum( (chain_mean-samp_mean)**2, axis=0 ) 
    W = (1./M) * np.sum(chain_var, axis=0)
    Vbar = (N-1)/N * W + (M+1)/(M*N) * B
    GR = Vbar/W
    
    with open(outf, 'w') as of:
        inds = int((N/np.mean(tau)))
        print("chains: ", chains.shape, file=of)
        print("tau: ", np.mean(tau), np.min(tau), np.max(tau), np.std(tau), file=of)
        print("burnin: %d, thin: %d" %(burnin, thin), file=of)
        print("independent samps per chain: %d" %inds, file=of)
        print("GR", file=of)
        print(GR, file=of)
        if np.any(GR > 1.1):
            print("FAILED GR", file=of)
        if inds < 50: 
            print("POTENTIALLY BAD TAU", file=of)
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels)
    return

for af in allfs:
    save_cleaned_chains(af)


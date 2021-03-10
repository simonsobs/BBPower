import os.path
import numpy as np
import glob
import emcee
from labels import alllabels

allfs = glob.glob('/mnt/zfsusers/mabitbol/BBPower/baseline_full/sim0*/output*/')
allfs.sort()

dones = []
for af in allfs:
    if os.path.isfile(af+'cleaned_chains.npz'):
        dones.append(af)
for df in dones:
    allfs.remove(df)


def save_cleaned_chains(fdir):
    outf = fdir+'chain_info.txt'
    reader = emcee.backends.HDFBackend(fdir+'param_chains.npz.h5')
    x = np.load(fdir+'param_chains.npz')
    labels = ['$%s$' %alllabels[k] for k in x['names']]

    try: 
        tau = reader.get_autocorr_time(tol=50)
        taust = 'good tau'
    except:
        taust = 'POTENTIALLY BAD TAU'
        tau = reader.get_autocorr_time(tol=0)
    burnin = int(5. * np.max(tau))
    thin = int(0.5 * np.max(tau))

    samples = reader.get_chain(discard=burnin, flat=True, thin=thin) 
    N = samples.shape[0]

    with open(outf, 'w') as of:
        print("N: ", N, file=of)
        print("burnin: %d, thin: %d" %(burnin, thin), file=of)
        print("tau:", file=of)
        print(tau, file=of)
        print(taust, file=of)
    np.savez(fdir+'cleaned_chains', samples=samples, names=x['names'], labels=labels)
    return

for af in allfs:
    save_cleaned_chains(af)



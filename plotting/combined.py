import os.path
import numpy as np
import glob
import emcee
from labels import alllabels

import matplotlib as mpl
from getdist import plots, MCSamples
from labels import alllabels, allranges

npf = 'hybrid_masked_10p'
allfs = glob.glob(f'/mnt/zfsusers/mabitbol/BBPower/{npf}/sim0*/output*/')
allfs.sort()


def save_cleaned_chains(fdir):
    reader = emcee.backends.HDFBackend(f'{fdir}param_chains.npz.h5')
    x = np.load(f'{fdir}param_chains.npz')
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

    with open(f'{fdir}chain_info.txt', 'w') as of:
        print("N: ", N, file=of)
        print("burnin: %d, thin: %d" %(burnin, thin), file=of)
        print("tau:", file=of)
        print(tau, file=of)
        print(taust, file=of)
    np.savez(f'{fdir}cleaned_chains', samples=samples, names=x['names'], labels=labels)
    return

for af in allfs:
    save_cleaned_chains(af)


def getdist_clean(fdir):
    sampler = np.load(f'{fdir}cleaned_chains.npz')
    vals = [allranges[k] for k in sampler['labels']]
    ranges = dict(zip(sampler['labels'], vals))
    gd_samps = MCSamples(samples=sampler['samples'], 
                         names=sampler['labels'], 
                         labels=[p.strip('$') for p in sampler['labels']], 
                         ranges=ranges)
    return gd_samps

for af in allfs: 
    sname = af.split('/')[-4] + af.split('/')[-3] + af.split('/')[-2]
    #sname = af.split('/')[-3]
    outf = f'{af}results.txt'

    samps = getdist_clean(af)
    samps.getTable(limit=1).write(outf)

    z = samps.paramNames.list()
    g = plots.get_subplot_plotter(width_inch=16)
    g.settings.axes_fontsize=14
    g.settings.axes_labelsize=14
    g.settings.line_styles = 'tab10'
    g.triangle_plot([samps], z, shaded=True, title_limit=1)
    mpl.pyplot.savefig(f'/mnt/zfsusers/mabitbol/notebooks/data_and_figures/{sname}_triangle')
    mpl.pyplot.close()


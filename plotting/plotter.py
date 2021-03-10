import numpy as np
import matplotlib as mpl
import glob
from getdist import plots, MCSamples
from labels import alllabels, allranges

allfs = glob.glob('/mnt/zfsusers/mabitbol/BBPower/baseline_full/sim0*/output*/')
allfs.sort()

def getdist_clean(fdir):
    sampler = np.load(fdir+'cleaned_chains.npz')
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
    outf = af+'results.txt'

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



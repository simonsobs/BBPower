import numpy as np
import sys

bbpower_dir = "/global/homes/k/kwolz/bbdev/BBPower"

sys.path.insert(0, f"{bbpower_dir}/examples")
from noise_calc import Simons_Observatory_V3_SA_beams  # noqa

outdir = "/pscratch/sd/k/kwolz/BBPower/examples/"
names = ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2']

larr = np.arange(10000)
beams = Simons_Observatory_V3_SA_beams(larr)

flist = f"{outdir}/data/beams_list.txt"
f = open(flist, "w")

for ib, bb in enumerate(beams):
    fname = f"{outdir}/data/beams/beam_{names[ib]}.txt"
    f.write(fname + "\n")
    np.savetxt(fname, np.transpose([larr, bb]))
f.close()
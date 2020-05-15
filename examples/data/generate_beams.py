from noise_calc import Simons_Observatory_V3_SA_beams
import numpy as np
import matplotlib.pyplot as plt

names=['LF1','LF2','MF1','MF2','UHF1','UHF2']

larr=np.arange(10000)
beams=Simons_Observatory_V3_SA_beams(larr)

for ib, bb in enumerate(beams):
    np.savetxt("data/beams/beam_"+names[ib]+'.txt',np.transpose([larr,bb]))
    plt.plot(larr,bb)
plt.show()

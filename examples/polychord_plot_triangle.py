import sys, os
import yaml
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1, f'bbpower')
from param_manager import ParameterManager

config_dir = f'test/test_out' # Contains "config_copy.yml"
base_dir = f'test/test_out/param_chains'
file_root = 'pch'

# Labels and true values
labdir = {'A_lens':     'A_{\\rm lens}',
          'r_tensor':   'r',
          'beta_d':     '\\beta_d',
          'epsilon_ds': '\\epsilon_{ds}',
          'alpha_d_bb': '\\alpha_d',
          'amp_d_bb':   'A_d',
          'beta_s':     '\\beta_s',
          'alpha_s_bb': '\\alpha_s',
          'amp_s_bb':   'A_s'}
truth = {'A_lens':      1.,
         'r_tensor':    0.,
         'beta_d':      1.59,
         'epsilon_ds':  0.,
         'alpha_d_bb':  -0.2,
         'amp_d_bb':    5.,
         'beta_s':      -3.,
         'alpha_s_bb':  -0.4,
         'amp_s_bb':    2.}


overall_config = yaml.load(open(f'{config_dir}/config_copy.yml'), Loader=yaml.FullLoader)
conf = overall_config.get('BBCompSep', {})
params = ParameterManager(conf)
prior = {n:pr for n, pr in zip(params.p_free_names, params.p_free_priors)}

# Create .paramnames file used by getdist
names = []
pfile = open(f'{base_dir}/{file_root}.paramnames', 'w')
for k,v in labdir.items():
    if k in params.p_free_names:
        names.append(k)
        pfile.write(f'{k:>{10}}  {v:>{13}} \n')
pfile.close()

# Make corner plot
samples = getdist.mcsamples.loadMCSamples(f'{base_dir}/{file_root}')
g = getdist.plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)
for i, n in enumerate(names):
    v = float(truth[n])
    g.subplots[i, i].plot([v, v], [0, 1], ls='-', color='r')
    for j in range(i + 1, len(names)):
        u = truth[names[j]]
        g.subplots[j, i].plot([v], [u], marker='o', color='r')
g.export(f'{base_dir}_triangle.pdf')



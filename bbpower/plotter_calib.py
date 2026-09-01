#!/usr/bin/env python
"""
Quick-look plots for a bbcalib run.

  * data vs. best-fit model, one panel per selected spectrum
  * residual pulls (data - model)/sigma, with the chi2 per spectrum
  * a triangle plot of the posterior, if a chain is present

Usage:
    python plot_calib.py --dir satp1/outputs/fiducial [--tag bestfit]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402


def load(dirname, tag):
    for t in ([tag] if tag else ['bestfit', 'minimize', 'single_point']):
        f = os.path.join(dirname, '%s.npz' % t)
        if os.path.exists(f):
            print("reading %s" % f)
            return np.load(f, allow_pickle=True), t
    raise FileNotFoundError("no bestfit/minimize/single_point npz in %s"
                            % dirname)


def spectra_slices(d):
    starts = d['spec_starts']
    ends = np.append(starts[1:], len(d['data']))
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def plot_spectra(d, tag, outdir, residuals=False):
    labels = [str(x) for x in d['spec_labels']]
    slices = spectra_slices(d)
    ells = d['spec_ells']
    err = np.sqrt(d['cov_diag'])
    n = len(labels)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                            squeeze=False)
    for k, (lab, sl) in enumerate(zip(labels, slices)):
        ax = axs[k // ncol][k % ncol]
        x, y, m, e = ells[sl], d['data'][sl], d['model'][sl], err[sl]
        if residuals:
            ax.axhline(0, color='0.6', lw=.8)
            for s, c in ((1, '0.85'), (2, '0.93')):
                ax.axhspan(-s, s, color=c, zorder=0)
            ax.errorbar(x, (y - m) / e, yerr=1., fmt='.', ms=3, lw=.8,
                        color='C3')
            ax.set_ylim(-4.5, 4.5)
            c2 = np.sum(((y - m) / e)**2)
            ax.set_title('%s\n$\\chi^2_{\\rm diag}$=%.0f/%d'
                         % (lab, c2, len(x)), fontsize=6.5)
        else:
            ax.errorbar(x, y, yerr=e, fmt='.', ms=3, lw=.8, color='k',
                        label='data')
            ax.plot(x, m, color='C3', lw=1.2, label='model')
            ax.set_title(lab, fontsize=6.5)
            if lab.startswith(('TT', 'EE')) and np.all(y[np.isfinite(y)] > 0):
                ax.set_yscale('log')
        ax.tick_params(labelsize=6)
        if k // ncol == nrow - 1:
            ax.set_xlabel(r'$\ell$', fontsize=7)
        if k % ncol == 0:
            ax.set_ylabel(r'$(\ell-m)/\sigma$' if residuals
                          else r'$D_\ell\ [\mu K^2]$', fontsize=7)
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].axis('off')
    fig.tight_layout()
    name = 'residuals' if residuals else 'spectra'
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, '%s_%s.%s' % (name, tag, ext)),
                    dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("  wrote %s_%s.png/pdf" % (name, tag))


def plot_triangle(chain_file, outdir):
    c = np.load(chain_file, allow_pickle=True)
    names = [str(x) for x in c['names']]
    chain = c['chain']
    n = len(names)
    fig, axs = plt.subplots(n, n, figsize=(1.35 * n, 1.35 * n), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axs[i][j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                ax.hist(chain[:, i], bins=45, color='C0', histtype='step')
                q = np.percentile(chain[:, i], [16, 50, 84])
                for v in q:
                    ax.axvline(v, color='C3', lw=.7, ls='--')
                ax.set_title('%s\n$%.4f^{+%.4f}_{-%.4f}$'
                             % (names[i], q[1], q[2] - q[1], q[1] - q[0]),
                             fontsize=6)
                ax.set_yticks([])
            else:
                h, xe, ye = np.histogram2d(chain[:, j], chain[:, i], bins=40)
                hs = np.sort(h.ravel())[::-1]
                cs = np.cumsum(hs) / hs.sum()
                lv = [hs[np.searchsorted(cs, p)] for p in (0.95, 0.68)]
                ax.contourf(0.5 * (xe[1:] + xe[:-1]),
                            0.5 * (ye[1:] + ye[:-1]), h.T,
                            levels=lv + [h.max() + 1],
                            colors=['#c6dbef', '#4292c6'])
            ax.tick_params(labelsize=5)
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=6)
            else:
                ax.set_yticklabels([])
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, 'triangle.%s' % ext), dpi=130,
                    bbox_inches='tight')
    plt.close(fig)
    print("  wrote triangle.png/pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    d, tag = load(args.dir, args.tag)
    ndof = int(d['n_data']) - int(d['ndim'])
    print("chi2 = %.2f  ndof = %d  chi2/ndof = %.3f"
          % (d['chi2'], ndof, d['chi2'] / ndof))
    print("parameters:")
    for n_, v in zip(d['free_names'], d['free_values']):
        print("   %-24s = %12.6f" % (n_, v))

    plot_spectra(d, tag, args.dir, residuals=False)
    plot_spectra(d, tag, args.dir, residuals=True)

    chain_file = os.path.join(args.dir, 'chain.npz')
    if os.path.exists(chain_file):
        plot_triangle(chain_file, args.dir)


if __name__ == '__main__':
    main()

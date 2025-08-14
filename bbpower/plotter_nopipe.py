import healpy as hp
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import dominate as dom
import dominate.tags as dtg
import sacc
import os
import yaml
import time
import bbpower.mpi_utils as mpi

from itertools import combinations_with_replacement as cwr

matplotlib.use('Agg')


def _yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


labels_dict = {
    'A_lens': '$A_{\\rm lens}$',
    'r_tensor': '$r$',
    'beta_d': '$\\beta_d$',
    'epsilon_ds': '$\\epsilon_{ds}$',
    'alpha_d_bb': '$\\alpha_d^{BB}$',
    'amp_d_bb': '$A_d^{BB}$',
    'alpha_d_ee': '$\\alpha_d^{EE}$',
    'amp_d_ee': '$A_d^{EE}$',
    'beta_s': '$\\beta_s$',
    'alpha_s_bb': '$\\alpha_s^{BB}$',
    'amp_s_bb': '$A_s^{BB}$',
    'alpha_s_ee': '$\\alpha_s^{EE}$',
    'amp_s_ee': '$A_s^{EE}$',
    'amp_d_beta': '$B_d^{BB}$',
    'gamma_d_beta': '$\\gamma_d^{BB}$',
    'amp_s_beta': '$B_s^{BB}$',
    'gamma_s_beta': '$\\gamma_s^{BB}$',
}


def read_cells_from_sacc(sacc_file, type, tracer_list, lmin, lmax,
                         return_cov=False):
    """
    """
    if isinstance(sacc_file, str):
        s = sacc.Sacc.load_fits(sacc_file)
    else:
        s = sacc_file
    cl_dict = {}
    clerr_dict = {}
    for tr1, tr2 in cwr(tracer_list, 2):
        if return_cov:
            l, cl, cov = s.get_ell_cl(type, tr1, tr2, return_cov=True)
        else:
            l, cl = s.get_ell_cl(type, tr1, tr2)
        msk = np.logical_and(l <= lmax, l >= lmin)
        l = l[msk]
        cl_dict[(tr1, tr2)] = cl[msk]
        if return_cov:
            clerr_dict[(tr1, tr2)] = np.sqrt(np.fabs(np.diag(cov)))[msk]

    if return_cov:
        return l, cl_dict, clerr_dict
    return l, cl_dict


def plot_triangle(lb, tracer_labels, clb_dict, clb_err_dict, clb_best_dict,
                  typ, text=None):
    """
    """
    dim = len(list(tracer_labels.keys()))
    typ_label = {"cl_bb": r"$D_\ell^{BB}$", "cl_eb": r"$D_\ell^{EB}$",
                 "cl_be": r"$D_\ell^{BE}$", "cl_ee": r"$D_\ell^{EE}$"} 
    fig, axes = plt.subplots(dim, dim,
                             figsize=(15,10),
                             constrained_layout=True,
                             sharex=True)
    for b1, t1 in enumerate(list(tracer_labels.keys())):
        for b2, t2 in enumerate(list(tracer_labels.keys())):
            if b2 < b1:
                axes[b2, b1].axis('off')
            else:
                label = f"{tracer_labels[t1]} x {tracer_labels[t2]}"
                ax = axes[b2, b1]
                x, y, yerr, ybest = (lb, clb_dict[(t1, t2)],
                                     clb_err_dict[(t1, t2)],
                                     clb_best_dict[(t1, t2)])
                simple_chi2 = np.sum((y - ybest)**2/yerr**2)
                ndof = len(y)
                ax.axhline(0, color="grey", linestyle="--")
                ax.plot([], [], ls="",
                        label=fr"$\chi^2/$ndof$={simple_chi2:.1f}/{ndof}$")
                ax.errorbar(x, y, yerr, color="navy", marker=".",
                            markerfacecolor="navy", linestyle="",
                            label=label)
                ax.plot(x, ybest, color="k")
                ax.legend(fontsize=9)
                if b1 == 0:
                    ax.set_ylabel(typ_label[typ], fontsize=15)
                if b2 == dim - 1:
                    ax.set_xlabel(r"$\ell$", fontsize=16)

    if text is not None:
        ax_upper = fig.add_subplot(333)
        ax_upper.plot([], [], color="w", label=text)
        ax_upper.axis('off')
        ax_upper.legend(fontsize=14)
    return fig, axes


class BBPlotter(object):
    """
    Plotting stage for BBPower. Plots power spectra, best-fit models, and
    posterior chains.
    """
    def __init__(self, args):
        """
        Initialize from the command line arguments.

        Parameters
        ----------
        args : str
            Command line arguments.
        """
        # Load the configuration file
        config = _yaml_loader(args.config)
        self.config = config["global"] | config["BBPlotter"]
        setattr(self, "data", self.config["data"])
        setattr(self, "config_fname", getattr(args, "config"))

        # Load global parameters
        self.nside = self.config['nside']
        self.npix = hp.nside2npix(self.nside)

    def create_page(self):
        """
        """
        # Open plots directory
        self.plot_dir = self.config["plot_dir"].format(sim_id=self.sim_id)
        if not os.path.isdir(self.plot_dir):
            os.mkdir(self.plot_dir)

        # Create HTML page
        self.doc = dom.document(title='BBPower plots page')
        with self.doc.head:
            dtg.link(rel='stylesheet', href='style.css')
            dtg.script(type='text/javascript', src='script.js')
        with self.doc:
            dtg.h1("Pipeline outputs")
            dtg.h2("Contents:", id='contents')
            lst = dtg.ul()
            lst += dtg.li(dtg.a('Bandpasses', href='#bandpasses'))
            lst += dtg.li(dtg.a('Coadded power spectra', href='#coadded'))
            if self.config['plot_cells_null']:
                lst += dtg.li(dtg.a('Null tests', href='#nulls'))
            if self.config['plot_likelihood']:
                lst += dtg.li(dtg.a('Likelihood', href='#like'))

    def add_bandpasses(self):
        """
        """
        with self.doc:
            dtg.h2("Bandpasses", id='bandpasses')
            lst = dtg.ul()

            # Overall plot
            title = 'Bandpasses summary'
            fname = self.plot_dir + '/bpass_summary.png'
            plt.figure()
            plt.title(title, fontsize=14)
            for n, t in self.s_cd_x.tracers.items():
                nu_mean = np.sum(t.bandpass*t.nu**3)/np.sum(t.bandpass*t.nu**2)
                plt.plot(t.nu, t.bandpass/np.amax(t.bandpass),
                         label=n+', $\\langle\\nu\\rangle=%.1lf\\,{\\rm GHz}$' % nu_mean)  # noqa
            plt.xlabel('$\\nu\\,[{\\rm GHz}]$', fontsize=14)
            plt.ylabel('Transmission', fontsize=14)
            plt.ylim([0., 1.3])
            plt.legend(frameon=0, ncol=2, labelspacing=0.1, loc='upper left')
            plt.xscale('log')
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            lst += dtg.li(dtg.a(title, href=fname))

            for n, t in self.s_cd_x.tracers.items():
                fname = self.plot_dir + '/bpass_' + n + '.png'
                plt.figure()
                plt.title(title, fontsize=14)
                plt.plot(t.nu, t.bandpass/np.amax(t.bandpass))
                plt.xlabel('$\\nu\\,[{\\rm GHz}]$', fontsize=14)
                plt.ylabel('Transmission', fontsize=14)
                plt.ylim([0., 1.05])
                plt.savefig(fname, bbox_inches='tight')
                plt.close()
                lst += dtg.li(dtg.a(title, href=fname))
            dtg.div(dtg.a('Back to TOC', href='#contents'))

    def add_coadded(self):
        (do_best, do_fid, do_cross,
         do_tot, do_noise) = (
            self.best_fit is not None,
            self.s_fid is not None,
            self.s_cd_x is not None,
            self.s_cd_t is not None,
            self.s_cd_n is not None
        )
        print("do_best", do_best)
        if do_best:
            best_fit_label = "\n".join(
                [fr"{ll}: {b}"for ll, b in self.best_fit.items()]
            )
        else:
            best_fit_label = ""
        if not do_cross:
            print("No cross-coadded spectra found. Do not plot any spectra.")
            return
        with self.doc:
            dtg.h2("Coadded power spectra", id='coadded')
            lst = dtg.ul()
            offset = 2
            for t1, t2 in self.s_cd_x.get_tracer_combinations():
                for p1, p2 in cwr(self.pols, 2):
                    x = p1 + p2
                    typ = 'cl_' + x
                    l_cro, cl_cro = self.s_cd_x.get_ell_cl(
                        typ, t1, t2, return_cov=False
                    )
                    l_cov, _, cov = self.s_cov.get_ell_cl(
                        typ, t1, t2, return_cov=True
                    )
                    msk_cov = np.logical_and(l_cov <= self.lmax,
                                             l_cov >= self.lmin)
                    if len(l_cro) == 0:
                        print(f"No data for {typ} found. Skip.")
                        continue

                    # Plot title
                    title = f"{t1} x {t2}, {typ}"
                    # Plot file
                    fname = self.plot_dir + '/cls_'
                    fname += f"{t1}_x_{t2}_{typ}.png"
                    f, (main, sub) = plt.subplots(
                        2, 1, sharex=True, figsize=(6, 4),
                        gridspec_kw={'height_ratios': [3, 1]}
                    )
                    f.suptitle(title, fontsize=14)
                    if do_fid:
                        l_fid, cl_fid = self.s_fid.get_ell_cl(typ, t1, t2)
                        msk_fid = np.logical_and(l_fid <= self.lmax,
                                                 l_fid >= self.lmin)
                        main.plot(l_fid[msk_fid], cl_fid[msk_fid], 'k--',
                                  label='Fiducial')
                    if do_best:
                        l_best, cl_best = self.s_best.get_ell_cl(
                            typ, t1, t2
                        )
                        msk_best = np.logical_and(l_best <= self.lmax,
                                                  l_best >= self.lmin)
                        main.plot(l_best[msk_best], cl_best[msk_best],
                                  'k-', label="Best fit")
                        main.plot([], [], "w-", label=best_fit_label)
                    sub.axhspan(-3, 3, facecolor="k", alpha=0.1)
                    sub.axhspan(-2, 2, facecolor="k", alpha=0.2)
                    sub.axhspan(-1, 1, facecolor="k", alpha=0.3)
                    sub.axhline(0, color="k")

                    if do_tot:
                        l_tot, cl_tot = self.s_cd_t.get_ell_cl(
                            typ, t1, t2
                        )
                        msk_tot = np.logical_and(l_tot <= self.lmax,
                                                 l_tot >= self.lmin)
                        x, y = (l_tot[msk_tot], cl_tot[msk_tot])
                        main.plot(
                            x - offset, y, label='Total coadd',
                            color="darkred", marker=".",
                            markerfacecolor='darkred', linestyle="",
                        )
                        eb_tot = main.plot(
                            x + offset, -y, color="darkred",
                            marker=".", markerfacecolor='white',
                            linestyle="",
                        )
                        eb_tot[-1][0].set_linestyle('--')

                    if do_noise:
                        l_noi, cl_noi = self.s_cd_n.get_ell_cl(
                            typ, t1, t2
                        )
                        msk_noi = np.logical_and(l_noi <= self.lmax,
                                                 l_noi >= self.lmin)
                        x, y = (l_noi[msk_noi], cl_noi[msk_noi])
                        main.plot(
                            x, y, label='Noise',
                            color="darkorange", marker=".",
                            markerfacecolor='darkorange', linestyle="",
                        )
                        eb_noi = main.plot(
                            x + offset, -y, color="darkorange",
                            marker=".", markerfacecolor="white",
                            linestyle="",
                        )
                        eb_noi[-1][0].set_linestyle('--')

                    msk_cro = np.logical_and(l_cro <= self.lmax,
                                             l_cro >= self.lmin)
                    x, y, yerr = (
                        l_cro[msk_cro], cl_cro[msk_cro],
                        np.sqrt(np.fabs(np.diag(cov)))[msk_cov]
                    )
                    main.errorbar(
                        x - offset, y, yerr, label='Cross coadd',
                        color="navy", marker=".",
                        markerfacecolor="navy", linestyle="",
                    )
                    eb_cro = main.errorbar(
                        x + offset, -y, yerr, color="navy", marker=".",
                        markerfacecolor="white", linestyle="",
                    )
                    eb_cro[-1][0].set_linestyle('--')

                    if do_best:
                        if do_fid:
                            sub.plot(
                                l_fid[msk_fid],
                                (cl_fid[msk_fid] - cl_best[msk_best])/yerr,
                                "k--"
                            )
                        sub.plot(
                            x, (y - cl_best[msk_best])/yerr,
                            color="navy", marker=".", markerfacecolor="navy",
                            linestyle=""
                        )
                        simple_chi2 = np.sum((y - cl_best[msk_best])**2/yerr**2)
                        ndof = len(y)
                        main.plot(
                            [], [], ls="",
                            label=fr"This panel: $\chi^2={simple_chi2:.1f}$ (ndof$={ndof}$)"  # noqa
                        )
                    sub.set_ylabel(
                        r"$(C_\ell-C_\ell^{\rm best})/\sigma(C_\ell)$"
                    )
                    sub.set_xlabel('$\\ell$', fontsize=15)
                    sub.set_ylim((-5, 5))
                    main.set_yscale('log')
                    if self.config['compute_dell']:
                        main.set_ylabel('$D_\\ell$', fontsize=15)
                    else:
                        main.set_ylabel('$C_\\ell$', fontsize=15)
                    main.legend(loc="upper left", bbox_to_anchor=(1, 1),
                                fontsize=12)
                    plt.savefig(fname, bbox_inches='tight')
                    plt.close()
                    lst += dtg.li(dtg.a(title, href=fname))

            # Plots C_ells triangle plots
            if not do_best:
                return
            for p1, p2 in cwr(self.pols, 2):
                tracer_labels = {
                    n: i+1
                    for i, n in enumerate(list(self.s_cd_x.tracers.keys()))
                }
                x = p1 + p2
                typ = 'cl_' + x
                lb, clb = read_cells_from_sacc(
                    self.s_cd_x, typ, list(tracer_labels.keys()),
                    self.lmin, self.lmax, return_cov=False
                )
                _, clb_best = read_cells_from_sacc(
                    self.s_best, typ, list(tracer_labels.keys()),
                    self.lmin, self.lmax, return_cov=False
                )
                _, _, err = read_cells_from_sacc(
                    self.s_cov, typ, list(tracer_labels.keys()),
                    self.lmin, self.lmax, return_cov=True
                )
            fig, axes = plot_triangle(
                lb, tracer_labels, clb, err, clb_best, typ,
                text="Best fit:\n"+best_fit_label
            )
            fig.savefig(f"{self.plot_dir}/cells_cross_{typ}.pdf",
                        bbox_inches="tight")
            print(f"PLOTS {self.plot_dir}")
            dtg.div(dtg.a('Back to TOC', href='#contents'))

    def add_nulls(self):
        """
        """
        do_nulls = self.s_null is not None
        if not do_nulls:
            return
        with self.doc:
            dtg.h2("Null tests", id='nulls')
            lst = dtg.ul()

            for t1, t2 in self.s_null.get_tracer_combinations():
                title = f"{t1} x {t2}"
                fname = self.plot_dir + '/cls_null_'
                fname += f"{t1}_x_{t2}.png"
                plt.figure()
                plt.title(title, fontsize=15)
                for p1 in range(2):
                    for p2 in range(2):
                        x = self.pols[p1] + self.pols[p2]
                        typ = 'cl_' + x
                        l, cl, cv = self.s_null.get_ell_cl(typ, t1, t2,
                                                           return_cov=True)
                        msk = l < self.lmax
                        el = np.sqrt(np.fabs(np.diag(cv)))[msk]
                        plt.errorbar(l[msk], cl[msk]/el,
                                     yerr=np.ones_like(el),
                                     fmt=self.cols_typ[x]+'-', label=x)
                plt.xlabel('$\\ell$', fontsize=15)
                plt.ylabel('$C_\\ell/\\sigma_\\ell$', fontsize=15)
                plt.legend()
                plt.savefig(fname, bbox_index='tight')
                plt.close()
                lst += dtg.li(dtg.a(title, href=fname))

            dtg.div(dtg.a('Back to TOC', href='#contents'))

    def add_contours(self):
        """
        """
        from getdist import MCSamples
        from getdist import plots as gplots

        with self.doc:
            dtg.h2("Likelihood", id='like')
            lst = dtg.ul()

            # Select only selected parameters for which we have labels
            names_common = list(set(list(self.chain['names']))
                                & self.config['truth'].keys()
                                & set(self.config['params_plot']))
            msk_common = np.array([n in names_common
                                   for n in self.chain['names']])
            _, nsamp, npar_chain = self.chain['chain'].shape
            chain = self.chain['chain'][:, nsamp//4:, :].reshape([-1, npar_chain])[:, msk_common]  # noqa
            names_common = np.array(self.chain['names'])[msk_common]
            labels = [labels_dict[n].replace("$", "") for n in names_common]

            # Getdist
            samples = MCSamples(
                samples=chain,
                names=names_common,
                labels=labels
            )

            # Plot posteriors
            g = gplots.getSubplotPlotter()
            g.settings.title_limit_fontsize = 12
            g.settings.axes_fontsize = 14
            g.settings.axes_labelsize = 18
            g.triangle_plot([samples],
                            filled=True,
                            contour_colors=['navy'],
                            title_limit=1)

            # Plot truth values
            for i, n in enumerate(names_common):
                v = self.config["truth"][n]
                if v is None:
                    continue
                g.subplots[i, i].plot([v, v], [0, 1], 'r-')
                for j in range(i + 1, len(names_common)):
                    u = self.config["truth"][names_common[j]]
                    g.subplots[j, i].plot([v], [u], 'r*')

            # Save
            fname = self.plot_dir + '/triangle.pdf'
            g.export(fname)
            lst += dtg.li(dtg.a("Likelihood contours", href=fname))

            dtg.div(dtg.a('Back to TOC', href='#contents'))

    def write_page(self):
        """
        """
        plots_page = self.config['plots_page'].format(sim_id=self.sim_id)
        with open(plots_page, 'w') as f:
            f.write(self.doc.render())

    def read_inputs(self):
        """
        """
        print("Reading inputs")

        # Power spectra
        saccs_dict = {
            "cells_fiducial": "s_fid",
            "cells_best_fit": "s_best",
            "cells_coadded": "s_cd_x",
            "cells_coadded_total": "s_cd_t",
            "cells_noise": "s_cd_n",
            "cells_null": "s_null",
            "cells_coadded_cov": "s_cov",
        }

        # Load relevant sacc files for plotting
        for cl, sc in saccs_dict.items():
            setattr(self, sc, None)
            if self.data[cl] is None:
                print(cl, "is None")
                continue
            data = self.data[cl].format(sim_id=self.sim_id)
            if not os.path.isfile(data):
                print(cl, "is not a file")
                continue
            if hasattr(self, f"plot_{cl}"):
                if not self.config[f"plot_{cl}"]:
                    print(cl, "not plotted")
                    continue
            setattr(self, sc, sacc.Sacc.load_fits(data))

        # Keep only desired tracers
        for s in saccs_dict.values():
            sc = getattr(self, s)
            if sc is None:
                continue
            tr_list = list(sc.tracers.keys())
            for t in tr_list:
                if t not in self.config["map_sets"]:
                    sc.remove_tracers([t])
            setattr(self, s, sc)

        # Chains
        chain_file = self.config["chain_file"].format(sim_id=self.sim_id)
        if not os.path.isfile(chain_file):
            self.config['plot_likelihood'] = False
            print("No posterior chain file found. Do not plot it.")
        if self.config['plot_likelihood']:
            self.chain = np.load(chain_file)

        # Best-fit parameters
        chi2_file = self.config["chi2_file"].format(sim_id=self.sim_id)
        self.best_fit, self.chisq, self.pte = (None, None, None)
        if os.path.isfile(chi2_file):
            chisq = np.load(chi2_file)
            self.best_fit = {
                labels_dict[n]: f"{float(p):.3f}"
                for n, p in zip(chisq["names"], chisq['params'])
            }
            self.best_fit["chi2"] = str(int(chisq['chi2']))
            self.best_fit["ndof"] = str(int(chisq['ndof']))
            self.best_fit["pte"] = f"{chisq['pte']:.1e}"

            self.chisq = chisq["chi2"]
            self.ndof = chisq["ndof"]
            self.pte = chisq["pte"]

        self.cols_typ = {'ee': 'r', 'eb': 'g', 'be': 'y', 'bb': 'b'}
        self.lmin = self.config['lmin_plot']
        self.lmax = self.config['lmax_plot']
        self.pols = [p.lower() for p in self.config['pol_channels']]

    def run(self):
        """
        Run the BB pipeline plotting stage.
        """
        self.read_inputs()
        self.create_page()
        self.add_bandpasses()
        self.add_coadded()
        if self.config['plot_cells_null']:
            self.add_nulls()
        if self.config['plot_likelihood']:
            self.add_contours()
        self.write_page()


def main(args):
    """
    Execute the BBPlotter stage with arguments args:
        * config: string.
          Path to configuration file with input parameters.
    """
    config = _yaml_loader(args.config)

    # Creating the simulation indices range to loop over
    sim_ids = config["global"]["sim_ids"]
    if isinstance(sim_ids, list):
        sim_ids = np.array(sim_ids, dtype=int)
    elif isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])
    else:
        sim_ids = [None]

    # MPI related initialization
    rank, size, comm = mpi.init(True)
    mpi_shared_list = sim_ids

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id in local_mpi_list:
        start = time.time()
        plotter = BBPlotter(args)
        setattr(plotter, "sim_id", sim_id)
        plotter.run()
    comm.Barrier()
    mpi.print_rnk0(f"Processed {len(sim_ids)} simulations "
                   f"in {time.time() - start:.1f} seconds.", rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plotting stage for BB Cross-CL pipeline"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to yaml file with pipeline configuration"
    )

    args = parser.parse_args()
    main(args)

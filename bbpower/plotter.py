from bbpipe import PipelineStage
from .data_types import FitsFile, DirFile, HTMLFile, NpzFile
import sacc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import dominate as dom
import dominate.tags as dtg
import os
from itertools import combinations_with_replacement as cwr

matplotlib.use('Agg')


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
}


class BBPlotter(PipelineStage):
    """
    Plotting stage for BBPower. Plots power spectra, best-fit models, and
    posterior chains.
    """
    name = "BBPlotter"
    inputs = [
        ('cells_coadded_total', FitsFile),
        ('cells_coadded', FitsFile),
        ('cells_noise', FitsFile),
        ('cells_null', FitsFile),
        ('cells_fiducial', FitsFile),
        ('cells_best_fit', FitsFile),
        ('chisq', NpzFile),
        ('param_chains', NpzFile)
    ]
    outputs = [
        ('plots', DirFile),
        ('plots_page', HTMLFile)
    ]
    config_options = {
        'pol_channels': ["e", "b"],
        'lmin_plot': 30,
        'lmax_plot': 300,
        'plot_coadded_total': True,
        'plot_noise': True,
        'plot_nulls': True,
        'plot_likelihood': True,
        'params_plot': None
    }

    def create_page(self):
        # Open plots directory
        if not os.path.isdir(self.get_output('plots')):
            os.mkdir(self.get_output('plots'))

        # Create HTML page
        self.doc = dom.document(title='BBPipe plots page')
        with self.doc.head:
            dtg.link(rel='stylesheet', href='style.css')
            dtg.script(type='text/javascript', src='script.js')
        with self.doc:
            dtg.h1("Pipeline outputs")
            dtg.h2("Contents:", id='contents')
            lst = dtg.ul()
            lst += dtg.li(dtg.a('Bandpasses', href='#bandpasses'))
            lst += dtg.li(dtg.a('Coadded power spectra', href='#coadded'))
            if self.config['plot_nulls']:
                lst += dtg.li(dtg.a('Null tests', href='#nulls'))
            if self.config['plot_likelihood']:
                lst += dtg.li(dtg.a('Likelihood', href='#like'))

    def add_bandpasses(self):
        with self.doc:
            dtg.h2("Bandpasses", id='bandpasses')
            lst = dtg.ul()
            # Overall plot
            title = 'Bandpasses summary'
            fname = self.get_output('plots') + '/bpass_summary.png'
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
                fname = self.get_output('plots') + '/bpass_' + n + '.png'
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
                    l_cro, cl_cro, cov_cro = self.s_cd_x.get_ell_cl(
                        typ, t1, t2, return_cov=True
                    )
                    if len(l_cro) == 0:
                        print(f"No data for {typ} found. Skip.")
                        continue

                    # Plot title
                    title = f"{t1} x {t2}, {typ}"
                    # Plot file
                    fname = self.get_output('plots') + '/cls_'
                    fname += f"{t1}_x_{t2}_{typ}.png"
                    print(fname)
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
                        l_tot, cl_tot, cov_tot = self.s_cd_t.get_ell_cl(
                            typ, t1, t2, return_cov=True
                        )
                        msk_tot = np.logical_and(l_tot <= self.lmax,
                                                 l_tot >= self.lmin)
                        x, y, yerr = (
                            l_tot[msk_tot], cl_tot[msk_tot],
                            np.sqrt(np.fabs(np.diag(cov_tot)))[msk_tot]
                        )
                        main.errorbar(
                            x - offset, y, yerr, label='Total coadd',
                            color="darkred", marker=".",
                            markerfacecolor='darkred', linestyle="",
                        )
                        eb_tot = main.errorbar(
                            x + offset, -y, yerr, color="darkred",
                            marker=".", markerfacecolor='white',
                            linestyle="",
                        )
                        eb_tot[-1][0].set_linestyle('--')
                        sub.plot(
                            x, (y - cl_best[msk_best])/yerr,
                            color="darkred", marker=".",
                            markerfacecolor="darkred", linestyle=""
                        )
                    if do_noise:
                        l_noi, cl_noi, cov_noi = self.s_cd_n.get_ell_cl(
                            typ, t1, t2, return_cov=True
                        )
                        msk_noi = np.logical_and(l_noi <= self.lmax,
                                                 l_noi >= self.lmin)
                        x, y, yerr = (l_noi[msk_noi], cl_noi[msk_noi],
                                        np.sqrt(np.fabs(np.diag(cov_noi)))[msk_noi])  # noqa
                        main.errorbar(
                            x, y, yerr, label='Noise',
                            color="darkorange", marker=".",
                            markerfacecolor='darkorange', linestyle="",
                        )
                        eb_noi = main.errorbar(
                            x + offset, -y, yerr, color="darkorange",
                            marker=".", markerfacecolor="white",
                            linestyle="",
                        )
                        eb_noi[-1][0].set_linestyle('--')
                        sub.plot(
                            x, (y - cl_best[msk_best])/yerr,
                            color="darkorange", marker=".",
                            markerfacecolor="darkorange", linestyle=""
                        )

                    msk_cro = np.logical_and(l_cro <= self.lmax,
                                             l_cro >= self.lmin)
                    x, y, yerr = (
                        l_cro[msk_cro], cl_cro[msk_cro],
                        np.sqrt(np.fabs(np.diag(cov_cro)))[msk_cro]
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
                fname = self.get_output('plots')+'/cls_null_'
                fname += f"{t1}_x_{t2}.png"
                print(fname)
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
        from getdist import MCSamples
        from getdist import plots as gplots

        with self.doc:
            dtg.h2("Likelihood", id='like')
            lst = dtg.ul()

            # TODO: we need to build this from the priors, I think.
            truth = {
                'A_lens': 1.,
                'r_tensor': 0.,
                'beta_d': 1.6,
                'epsilon_ds': 0.,
                'alpha_d_bb': -0.42,
                'amp_d_bb': 37.5,
                'beta_s': -3.,
                'alpha_s_bb': 1.2,
                'amp_s_bb': 5.
            }

            # Select only selected parameters for which we have labels
            names_common = list(set(list(self.chain['names']))
                                & truth.keys()
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
            g = gplots.getSubplotPlotter()
            g.settings.title_limit_fontsize = 12
            g.settings.axes_fontsize = 14
            g.settings.axes_labelsize = 18
            g.triangle_plot([samples],
                            filled=True,
                            contour_colors=['navy'],
                            title_limit=1)

            # for i, n in enumerate(names_common):
            #     v = truth[n]
            #     g.subplots[i,i].plot([v,v],[0,1],'r-')
            #     for j in range(i + 1, len(names_common)):
            #         u = truth[names_common[j]]
            #         g.subplots[j,i].plot([v],[u],'ro')

            # Save
            fname = self.get_output('plots') + '/triangle.pdf'
            g.export(fname)
            lst += dtg.li(dtg.a("Likelihood contours", href=fname))

            dtg.div(dtg.a('Back to TOC', href='#contents'))

    def write_page(self):
        with open(self.get_output('plots_page'), 'w') as f:
            f.write(self.doc.render())

    def read_inputs(self):
        print("Reading inputs")

        # Power spectra
        (self.s_fid, self.s_best, self.s_cd_x,
         self.s_cd_t, self.s_cd_n, self.s_null) = (
            None, None, None, None, None, None
        )
        if os.path.isfile(self.get_input('cells_fiducial')):
            self.s_fid = sacc.Sacc.load_fits(self.get_input('cells_fiducial'))
        if os.path.isfile(self.get_input('cells_best_fit')):
            self.s_best = sacc.Sacc.load_fits(self.get_input('cells_best_fit'))
        if os.path.isfile(self.get_input('cells_coadded')):
            self.s_cd_x = sacc.Sacc.load_fits(self.get_input('cells_coadded'))
        if (self.config['plot_coadded_total'] and os.path.isfile(self.get_input('cells_coadded_total'))):  # noqa
            self.s_cd_t = sacc.Sacc.load_fits(
                self.get_input('cells_coadded_total')
            )
        if (self.config['plot_noise'] and os.path.isfile(self.get_input('cells_noise'))):  # noqa
            self.s_cd_n = sacc.Sacc.load_fits(
                self.get_input('cells_noise')
            )
        if (self.config['plot_nulls'] and os.path.isfile(self.get_input('cells_noise'))):  # noqa
            self.s_null = sacc.Sacc.load_fits(
                self.get_input('cells_null')
            )

        # Keep only desired tracers
        for s in [self.s_fid, self.s_best, self.s_cd_x,
                  self.s_cd_t, self.s_cd_n, self.s_null]:
            if s is None:
                continue
            tr_list = list(s.tracers.keys())
            for t in tr_list:
                if t not in self.config["map_sets"]:
                    s.remove_tracers([t])

        # Chains
        if not os.path.isfile(self.get_input('param_chains')):
            self.config['plot_likelihood'] = False
            print("No posterior chain file found. Do not plot it.")
        if self.config['plot_likelihood']:
            self.chain = np.load(self.get_input('param_chains'))

        # Best-fit parameters
        self.best_fit, self.chisq, self.pte = (None, None, None)
        if os.path.isfile(self.get_input('chisq')):
            chisq = np.load(self.get_input('chisq'))
            self.best_fit = {
                labels_dict[n]: f"{float(p):.3f}"
                for n, p in zip(chisq["names"], chisq['params'])
            }
            self.best_fit["chisq"] = str(int(chisq['chisq']))
            self.best_fit["ndof"] = str(int(chisq['ndof']))
            self.best_fit["pte"] = f"{chisq['pte']:.1e}"

            self.chisq = chisq["chisq"]
            self.ndof = chisq["ndof"]
            self.pte = chisq["pte"]

        self.cols_typ = {'ee': 'r', 'eb': 'g', 'be': 'y', 'bb': 'b'}
        self.lmin = self.config['lmin_plot']
        self.lmax = self.config['lmax_plot']
        self.pols = [p.lower() for p in self.config['pol_channels']]

    def run(self):
        self.read_inputs()
        self.create_page()
        # self.add_bandpasses()
        self.add_coadded()
        if self.config['plot_nulls']:
            self.add_nulls()
        if self.config['plot_likelihood']:
            self.add_contours()
        self.write_page()


if __name__ == '__main_':
    cls = PipelineStage.main()

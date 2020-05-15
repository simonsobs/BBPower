from bbpipe import PipelineStage
from .types import TextFile, FitsFile, DirFile, HTMLFile, NpzFile
import sacc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dominate as dom
import dominate.tags as dtg
import os

class BBPlotter(PipelineStage):
    name="BBPlotter"
    inputs=[('cells_coadded_total', FitsFile), ('cells_coadded', FitsFile),
            ('cells_noise', FitsFile), ('cells_null', FitsFile),
            ('cells_fiducial', FitsFile), ('param_chains',NpzFile)]
    outputs=[('plots',DirFile), ('plots_page',HTMLFile)]
    config_options={'lmax_plot':300, 'plot_coadded_total': True,
                    'plot_noise': True, 'plot_nulls': True,
                    'plot_likelihood': True}

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
            dtg.h2("Contents:",id='contents')
            lst=dtg.ul()
            lst+=dtg.li(dtg.a('Bandpasses',href='#bandpasses'))
            lst+=dtg.li(dtg.a('Coadded power spectra',href='#coadded'))
            if self.config['plot_nulls']:
                lst+=dtg.li(dtg.a('Null tests',href='#nulls'))
            if self.config['plot_likelihood']:
                lst+=dtg.li(dtg.a('Likelihood',href='#like'))

    def add_bandpasses(self):
        with self.doc:
            dtg.h2("Bandpasses",id='bandpasses')
            lst=dtg.ul()
            # Overall plot
            title='Bandpasses summary'
            fname=self.get_output('plots')+'/bpass_summary.png'
            plt.figure()
            plt.title(title,fontsize=14)
            for n, t in self.s_fid.tracers.items():
                nu_mean=np.sum(t.bandpass*t.nu**3)/np.sum(t.bandpass*t.nu**2)
                plt.plot(t.nu,t.bandpass/np.amax(t.bandpass),label=n+', $\\langle\\nu\\rangle=%.1lf\\,{\\rm GHz}$'%nu_mean)
            plt.xlabel('$\\nu\\,[{\\rm GHz}]$',fontsize=14)
            plt.ylabel('Transmission',fontsize=14)
            plt.ylim([0.,1.3])
            plt.legend(frameon=0,ncol=2,labelspacing=0.1,loc='upper left')
            plt.xscale('log')
            plt.savefig(fname,bbox_inches='tight')
            plt.close()
            lst+=dtg.li(dtg.a(title,href=fname))

            for n, t in self.s_fid.tracers.items():
                title='Bandpass '+n
                fname=self.get_output('plots')+'/bpass_'+n+'.png'
                plt.figure()
                plt.title(title,fontsize=14)
                plt.plot(t.nu,t.bandpass/np.amax(t.bandpass))
                plt.xlabel('$\\nu\\,[{\\rm GHz}]$',fontsize=14)
                plt.ylabel('Transmission',fontsize=14)
                plt.ylim([0.,1.05])
                plt.savefig(fname,bbox_inches='tight')
                plt.close()
                lst+=dtg.li(dtg.a(title,href=fname))
            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def add_coadded(self):
        with self.doc:
            dtg.h2("Coadded power spectra",id='coadded')
            lst=dtg.ul()
            pols = ['e', 'b']
            print(self.s_fid.tracers)
            for t1, t2 in self.s_cd_x.get_tracer_combinations():
                for p1 in range(2):
                    if t1==t2:
                        p2range = range(p1, 2)
                    else:
                        p2range = range(2)
                    for p2 in p2range:
                        x = pols[p1] + pols[p2]
                        typ = 'cl_' + x
                        # Plot title
                        title = f"{t1} x {t2}, {typ}"
                        # Plot file
                        fname =self.get_output('plots')+'/cls_'
                        fname+= f"{t1}_x_{t2}_{typ}.png"
                        print(fname)
                        plt.figure()
                        plt.title(title, fontsize=14)
                        l, cl = self.s_fid.get_ell_cl(typ, t1, t2)
                        plt.plot(l[l<self.lmx], cl[l<self.lmx], 'k-', label='Fiducial model')
                        if self.config['plot_coadded_total']:
                            l, cl, cov = self.s_cd_t.get_ell_cl(typ, t1, t2, return_cov=True)
                            msk = l<self.lmx
                            el = np.sqrt(np.fabs(np.diag(cov)))[msk]
                            plt.errorbar(l[msk], cl[msk], yerr=el, fmt='ro',
                                         label='Total coadd')
                            eb=plt.errorbar(l[msk]+1, -cl[msk], yerr=el, fmt='ro', mfc='white')
                            eb[-1][0].set_linestyle('--')
                        if self.config['plot_noise']:
                            l, cl, cov = self.s_cd_n.get_ell_cl(typ, t1, t2, return_cov=True)
                            msk = l<self.lmx
                            el = np.sqrt(np.fabs(np.diag(cov)))[msk]
                            plt.errorbar(l[msk], cl[msk], yerr=el, fmt='yo',
                                         label='Noise')
                            eb=plt.errorbar(l[msk]+1, -cl[msk], yerr=el, fmt='yo', mfc='white')
                            eb[-1][0].set_linestyle('--')
                        l, cl, cov = self.s_cd_x.get_ell_cl(typ, t1, t2, return_cov=True)
                        msk = l<self.lmx
                        el = np.sqrt(np.fabs(np.diag(cov)))[msk]
                        plt.errorbar(l[msk], cl[msk], yerr=el, fmt='bo',
                                     label='Cross-coadd')
                        eb=plt.errorbar(l[msk]+1, -cl[msk], yerr=el, fmt='bo', mfc='white')
                        eb[-1][0].set_linestyle('--')
                        plt.yscale('log')
                        plt.xlabel('$\\ell$',fontsize=15)
                        if self.config['compute_dell']:
                            plt.ylabel('$D_\\ell$',fontsize=15)
                        else:
                            plt.ylabel('$C_\\ell$',fontsize=15)
                        plt.legend()
                        plt.savefig(fname,bbox_inches='tight')
                        plt.close()
                        lst+=dtg.li(dtg.a(title,href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))
                
    def add_nulls(self):
        with self.doc:
            dtg.h2("Null tests",id='nulls')
            lst=dtg.ul()

            pols = ['e', 'b']
            for t1, t2 in self.s_null.get_tracer_combinations():
                title = f"{t1} x {t2}"
                fname =self.get_output('plots')+'/cls_null_'
                fname+= f"{t1}_x_{t2}.png"
                print(fname)
                plt.figure()
                plt.title(title,fontsize=15)
                for p1 in range(2):
                    for p2 in range(2):
                        x = pols[p1] + pols[p2]
                        typ='cl_'+x
                        l, cl, cv = self.s_null.get_ell_cl(typ, t1, t2, return_cov=True)
                        msk = l<self.lmx
                        el = np.sqrt(np.fabs(np.diag(cv)))[msk]
                        plt.errorbar(l[msk], cl[msk]/el,
                                     yerr=np.ones_like(el),
                                     fmt=self.cols_typ[x]+'-', label=x)
                plt.xlabel('$\\ell$',fontsize=15)
                plt.ylabel('$C_\\ell/\\sigma_\\ell$',fontsize=15)
                plt.legend()
                plt.savefig(fname,bbox_index='tight')
                plt.close()
                lst+=dtg.li(dtg.a(title,href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def add_contours(self):
        from getdist import MCSamples
        from getdist import plots as gplots

        with self.doc:
            dtg.h2("Likelihood",id='like')
            lst=dtg.ul()

            # Labels and true values
            labdir={'A_lens':'A_{\\rm lens}',
                    'r_tensor':'r',
                    'beta_d':'\\beta_d',
                    'epsilon_ds':'\\epsilon_{ds}',
                    'alpha_d_bb':'\\alpha_d',
                    'amp_d_bb':'A_d',
                    'beta_s':'\\beta_s',
                    'alpha_s_bb':'\\alpha_s',
                    'amp_s_bb':'A_s'}
            # TODO: we need to build this from the priors, I think.
            truth={'A_lens':1.,
                   'r_tensor':0.,
                   'beta_d':1.59,
                   'epsilon_ds':0.,
                   'alpha_d_bb':-0.42,
                   'amp_d_bb':5.,
                   'beta_s':-3.,
                   'alpha_s_bb':-0.6,
                   'amp_s_bb':2.}

            # Select only parameters for which we have labels
            names_common=list(set(list(self.chain['names'])) & truth.keys())
            msk_common=np.array([n in names_common for n in self.chain['names']])
            npar=len(names_common)
            nwalk,nsamp,npar_chain=self.chain['chain'].shape
            chain=self.chain['chain'][:,nsamp//4:,:].reshape([-1,npar_chain])[:,msk_common]
            names_common=np.array(self.chain['names'])[msk_common]

            # Getdist
            samples=MCSamples(samples=chain,
                              names=names_common,
                              labels=[labdir[n] for n in names_common])
            g = gplots.getSubplotPlotter()
            g.triangle_plot([samples], filled=True)
            for i,n in enumerate(names_common):
                v=truth[n]
                g.subplots[i,i].plot([v,v],[0,1],'r-')
                for j in range(i+1,npar):
                    u=truth[names_common[j]]
                    g.subplots[j,i].plot([v],[u],'ro')

            # Save
            fname=self.get_output('plots')+'/triangle.png'
            g.export(fname)
            lst+=dtg.li(dtg.a("Likelihood contours",href=fname))

            dtg.div(dtg.a('Back to TOC',href='#contents'))

    def write_page(self):
        with open(self.get_output('plots_page'),'w') as f:
            f.write(self.doc.render())

    def read_inputs(self):
        print("Reading inputs")
        # Power spectra
        self.s_fid=sacc.Sacc.load_fits(self.get_input('cells_fiducial'))
        self.s_cd_x=sacc.Sacc.load_fits(self.get_input('cells_coadded'))
        if self.config['plot_coadded_total']:
            self.s_cd_t=sacc.Sacc.load_fits(self.get_input('cells_coadded_total'))
        if self.config['plot_noise']:
            self.s_cd_n=sacc.Sacc.load_fits(self.get_input('cells_noise'))
        if self.config['plot_nulls']:
            self.s_null=sacc.Sacc.load_fits(self.get_input('cells_null'))
        # Chains
        if self.config['plot_likelihood']:
            self.chain=np.load(self.get_input('param_chains'))

        self.cols_typ={'ee':'r','eb':'g','be':'y','bb':'b'}
        self.lmx = self.config['lmax_plot']

    def run(self):
        self.read_inputs()
        self.create_page()
        self.add_bandpasses()
        self.add_coadded()
        if self.config['plot_nulls']:
            self.add_nulls()
        if self.config['plot_likelihood']:
            self.add_contours()
        self.write_page()

if __name__ == '__main_':
    cls = PipelineStage.main()

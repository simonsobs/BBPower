import numpy as np
import matplotlib.pyplot as plt
import sacc
from itertools import combinations_with_replacement as cwr
import os

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
            el, cl, cov = s.get_ell_cl(type, tr1, tr2, return_cov=True)
        else:
            el, cl = s.get_ell_cl(type, tr1, tr2)
        msk = np.logical_and(el <= lmax, el >= lmin)
        el = el[msk]
        cl_dict[(tr1, tr2)] = cl[msk]
        if return_cov:
            clerr_dict[(tr1, tr2)] = np.sqrt(np.fabs(np.diag(cov)))[msk]

    if return_cov:
        return el, cl_dict, clerr_dict
    return el, cl_dict


def plot_triangle(lb, tracer_labels, clb_dict, clb_best_dict,
                  typ, c=None,clb_err_dict=None, text=None, dl2cl=False):
    """
    """
    dim = len(list(tracer_labels.keys()))
    pre = r"$D$" if not dl2cl else r"$C$"
    typ_label = {
        "cl_bb": pre+r"$_\ell^{BB}$",
        "cl_eb": pre+r"$_\ell^{EB}$",
        "cl_be": pre+r"$_\ell^{BE}$",
        "cl_ee": pre+r"$_\ell^{EE}$"
    }
    fig, axes = plt.subplots(dim, dim,
                             figsize=(15, 10),
                             constrained_layout=True,
                             sharex=True)
    for b1, t1 in enumerate(list(tracer_labels.keys())):
        for b2, t2 in enumerate(list(tracer_labels.keys())):
            if b2 < b1:
                axes[b2, b1].axis('off')
            else:
                label = f"{tracer_labels[t1]} x {tracer_labels[t2]}"
                ax = axes[b2, b1]
                fac = 1 if not dl2cl else 1./lb/(lb+1)*2.*np.pi
                x, y, ybest = (lb, fac*clb_dict[(t1, t2)],
                               fac*clb_best_dict[(t1, t2)])
                ndof = len(y)
                ax.axhline(0, color="grey", linestyle="--")

                if clb_err_dict is not None:
                    yerr = fac*clb_err_dict[(t1, t2)]
                    simple_chi2 = np.sum((y - ybest)**2/yerr**2)
                    ax.plot(
                        [], [], ls="",
                        label=fr"$\chi^2/$ndof$={simple_chi2:.1f}/{ndof}$",
                        c = c
                    )
                    ax.errorbar(x, y, yerr, color=c, marker=".",
                                markerfacecolor=c, linestyle="",
                                label=label)
                else:
                    ax.plot(x, y, color=c, marker=".",
                            markerfacecolor=c, linestyle="",
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


def plot_spectra(cells_coadded, cells_best_fit, cells_coadded_cov,
                 sim_ids, plot_dir, lmin=30, lmax=300):
    """
    Plot BB power spectra from data and best fits
    """ 
    sacc_cross = sacc.Sacc.load_fits(
        cells_coadded.format(sim_id=sim_ids[0])
    ) 
    tracer_labels = {
        n: i+1
        for i, n in enumerate(list(sacc_cross.tracers.keys()))
    }
    typ = 'cl_bb'

    clb_all = []
    clb_all_best = []
    chi2 = []
    pte = []

    for sim_id in sim_ids:
        print(sim_id)
        chi2 += [np.load(chi2_file.format(sim_id=sim_id))["chi2"]]
        ndof = np.load(chi2_file.format(sim_id=sim_id))["ndof"]
        pte += [np.load(chi2_file.format(sim_id=sim_id))["pte"]]
        sacc_cross = cells_coadded.format(sim_id=sim_id)
        sacc_best = cells_best_fit.format(sim_id=sim_id)
        _, clb = read_cells_from_sacc(
            sacc_cross, "cl_bb", list(tracer_labels.keys()),
            lmin, lmax, return_cov=False
        )
        _, clb_best = read_cells_from_sacc(
            sacc_best, "cl_bb", list(tracer_labels.keys()),
            lmin, lmax, return_cov=False
        )
        clb_all.append(clb)
        clb_all_best.append(clb_best)

    chi2, pte = (np.array(chi2), np.array(pte))
    chi2med = np.median(chi2)
    chi2up = np.percentile(chi2, 95) - chi2med
    chi2lo = chi2med - np.percentile(chi2, 5)
    ptemed = np.median(pte)
    pteup = np.percentile(pte, 95) - ptemed
    ptelo = ptemed - np.percentile(pte, 5)
    chi2_lab = fr"$\chi^2/$dof $= {int(np.median(chi2))}^{{+{int(chi2up)}}}_{{-{int(chi2lo)}}}\,(^{{68\%}}_{{32\%}}) / {int(ndof)}$" + "\n"
    chi2_lab += fr"PTE$={ptemed:.1E}^{{+{pteup:.1E}}}_{{-{ptelo:.1E}}}\,(^{{68\%}}_{{32\%}}$"

    lb, _, clb_err = read_cells_from_sacc(
        cells_coadded_cov, typ, list(tracer_labels.keys()),
        lmin, lmax, return_cov=True
    )

    clb_mean = {
        tr_pair: np.mean(
            np.array([clb_all[i][tr_pair] for i in range(len(sim_ids))]),
            axis=0
        ) for tr_pair in clb_all[sim_ids[0]]
    }
    clb_std = {
        tr_pair: np.std(
            np.array([clb_all[i][tr_pair] for i in range(len(sim_ids))]),
            axis=0
        ) for tr_pair in clb_all[sim_ids[0]]
    }
    clb_mean_std = {
        tr_pair: np.std(
            np.array([clb_all[i][tr_pair] for i in range(len(sim_ids))]),
            axis=0
        ) / np.sqrt(len(sim_ids))
        for tr_pair in clb_all[sim_ids[0]]
    }
    clb_mean_best = {
        tr_pair: np.mean(
            np.array([clb_all_best[i][tr_pair] for i in range(len(sim_ids))]),
            axis=0
        ) for tr_pair in clb_all[sim_ids[0]]
    }
    clb_mean_bias = {
        tr_pair: np.mean(
            np.array([clb_all[i][tr_pair] - clb_all_best[i][tr_pair]
                      for i in range(len(sim_ids))]),
            axis=0
        ) for tr_pair in clb_all[sim_ids[0]]
    }
    clb_std_bias = {
        tr_pair: np.std(
            np.array([clb_all[i][tr_pair] - clb_all_best[i][tr_pair]
                      for i in range(len(sim_ids))]),
            axis=0
        ) / np.sqrt(len(sim_ids))
        for tr_pair in clb_all[sim_ids[0]]
    }
    clb_zeros = {
        tr_pair: 0*clb_all[sim_ids[0]][tr_pair]
        for tr_pair in clb_all[sim_ids[0]]
    }

    for dl2cl in [True, False]:
        lab = "Dells" if not dl2cl else "Cells"
        fig, axes = plot_triangle(
            lb, tracer_labels, clb_mean, clb_mean_best, typ,
            clb_err_dict=clb_mean_std, text="Mean spectra\n"+chi2_lab,
            dl2cl=dl2cl,
            c=col
        )
        fig.suptitle(priors[idx])
        fig.savefig(f"{plot_dir}/{lab}_mean.png", bbox_inches="tight")

        fig, axes = plot_triangle(
            lb, tracer_labels, clb_mean_bias, clb_zeros, typ,
            clb_err_dict=clb_std_bias,
            text="Mean diff spectrum vs best fit\n"+chi2_lab,
            dl2cl=dl2cl,
            c=col
        )
        fig.suptitle(priors[idx])
        fig.savefig(f"{plot_dir}/{lab}_bias_mean.png", bbox_inches="tight")

        fig, axes = plot_triangle(
            lb, tracer_labels, clb_std, clb_err, typ,
            text="Simulation scatter vs covariance\n"+chi2_lab, dl2cl=dl2cl,
            c=col
        )
        fig.suptitle(priors[idx])
        fig.savefig(f"{plot_dir}/{lab}_std.png", bbox_inches="tight")
    print(f"PLOTS {plot_dir}")

## MAIN


priors = ["Jeffreys Priors", "Non-Jeffreys Priors"]
prior_filename = ["jeffreys_priors", "modified_azzoni_priors"]
colours = ["darkorange","navy"]

for idx in range(2):
    prior_fnam = prior_filename[idx]
    # Replace the file names by yours
    cells_coadded = "/mnt/users/eastonf/Jeffreys/BBPower/jeffreys-prior/data/cells_coadded_{sim_id:04d}.fits"
    cells_best_fit = "/mnt/users/eastonf/Jeffreys/BBPower/chains/all_channels/gaussian_fgs/fiducial_model_full/"+ prior_fnam +"/{sim_id:04d}/cells_model.fits"
    cells_coadded_cov = "/mnt/users/eastonf/Jeffreys/BBPower/jeffreys-prior/data/cells_coadded_cov_r0_Alens1_baseline_optimistic.fits"
    chi2_file = "/mnt/users/eastonf/Jeffreys/BBPower/chains/all_channels/gaussian_fgs/fiducial_model_full/"+ prior_fnam +"/{sim_id:04d}/chi2.npz"
    sim_ids = [i for i in range(100)]
    plot_dir = "/mnt/users/eastonf/Jeffreys/BBPower/comparison_plots/fiducial_model_full/modified_azzoni_priors/goodness_of_fit/" + prior_fnam
    col = colours[idx]

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_spectra(
        cells_coadded, cells_best_fit, cells_coadded_cov, sim_ids, plot_dir
    )
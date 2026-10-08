import yaml
import sacc
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def _yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def main(args):
    """
    Plot power spectra and per-bandpower parameter estimates for the
    per-bandpower model of BBPower.
    """
    config = _yaml_loader(args.config)
    plot_dir = f"{config['chains_dir']}/plots"
    if not os.path.isdir(plot_dir):
        print(f"Making directory {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)

    # Plot best-fit C_ells
    map_sets = list(config["global"]["map_sets"].keys())
    nbands = len(map_sets)
    fn_sacc = config['global']['data']['cells_coadded']

    fig, axes = plt.subplots(len(map_sets), 1, figsize=(5, 2.5*nbands))
    for i_map_set in range(nbands):
        ax = axes[i_map_set]
        tr = map_sets[i_map_set]

        l_data, cl_data, cov_data = sacc.Sacc.load_fits(fn_sacc).get_ell_cl(
            "cl_bb", tr, tr, return_cov=True)
        msk_data = np.logical_and(l_data <= config["BBCompSep"]["l_max"],
                                  l_data >= config["BBCompSep"]["l_min"])
        l_data = l_data[msk_data]
        cl_data = cl_data[msk_data]
        cov_data = cov_data[msk_data][:, msk_data]

        err_data = np.sqrt(np.diag(cov_data))
        n_bpws = len(l_data)

        for comp in ["cmb", "synch", "dust", "all"]:
            label = "" if comp == "all" else f"_{comp}"
            ls = ":" if comp == "all" else "-"
            fn_sacc_fiducial = f"{config['chains_dir']}/cls_fid.fits"
            fn_sacc_best_fit = f"{config['chains_dir']}/cells_model{label}.fits"  # noqa: E501
            s = sacc.Sacc.load_fits(fn_sacc_best_fit)
            lb, db = s.get_ell_cl("cl_bb", tr, tr)
            ax.plot(lb, db, label=comp, ls=ls)
            if comp == "all":
                ax.errorbar(l_data, cl_data, err_data, color="k", ls="",
                            marker=".", label="data")
                if os.path.isfile(fn_sacc_fiducial):
                    s_fid = sacc.Sacc.load_fits(fn_sacc_fiducial)
                    lb_fid, db_fid = s_fid.get_ell_cl(
                        "cl_bb", tr, tr)
                    lmin = config["BBPlotter"]["lmin_plot"]
                    lmax = config["BBPlotter"]["lmax_plot"]
                    lmsk = np.logical_and(lb_fid >= lmin, lb_fid <= lmax)
                    ax.plot(lb_fid[lmsk], db_fid[lmsk], "k--",
                            label="fiducial")
        ax.set_title(tr, y=0.9, va="top", fontsize=9)
        ax.set_ylabel(r"$D^{BB}_\ell$")
    axes[-1].set_xlabel(r"$\ell$")
    axes[0].legend(frameon=False, bbox_to_anchor=(1.03, 1), loc='upper left')
    fig.align_ylabels()

    plt.suptitle("Component-wise best-fit spectra",
                 fontsize=16, fontweight="bold", y=0.92)
    plt.savefig(f"{plot_dir}/ellwise_best_fit.pdf", bbox_inches="tight")
    print(f"{plot_dir}/ellwise_best_fit.pdf")

    # Plot per-ell parameters
    chains_fn = config["chains_dir"] + "/emcee_bpw{i_bpw:02}.npz"
    chains_0 = np.load(chains_fn.format(i_bpw=1))
    npar = len(chains_0["names"])

    fig, axes = plt.subplots(npar, nbands, figsize=(3*nbands, npar*2),
                             sharex=True)

    for i_map_set in range(nbands):
        tr = map_sets[i_map_set]
        chains = []

        for i_bpw in range(n_bpws):
            chains += [np.load(chains_fn.format(i_bpw=i_bpw))]

        for ipar, par in enumerate(chains_0["names"]):
            samps = np.array([chains[i_bpw]["chain"][..., ipar].flatten()
                              for i_bpw in range(n_bpws)])
            mean = np.mean(samps, axis=-1)
            std = np.std(samps, axis=-1)
            axes[ipar, i_map_set].errorbar(
                l_data, mean, std, marker=".", ls="",
                c=plt.get_cmap("tab10")(ipar))
            if i_map_set == 0:
                axes[ipar, i_map_set].set_ylabel(par)
            axes[ipar, i_map_set].set_title(tr, y=0.9, va="top", fontsize=7)
        axes[-1, i_map_set].set_xlabel(r"$\ell$")

    fig.align_ylabels()
    plt.suptitle("Bandpower-wise frequency model parameters",
                 fontsize=16, fontweight="bold", y=0.92)

    plt.savefig(f"{plot_dir}/ellwise_parameters.pdf", bbox_inches="tight")
    print(f"{plot_dir}/ellwise_parameters.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        help="Path to yaml file with pipeline configuration"
    )

    args = parser.parse_args()
    main(args)

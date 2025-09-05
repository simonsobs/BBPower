import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples
from getdist import plots as gplots
import os



### Variables to be set by the user ###

# Set to true if you want to plot histograms / boxplots
plot_hists = False

# Set to true if you want to plot triangle plots with both sets of data (plots triangles for dataruns 0000, 0010, 0020, ... 0090)
plot_triangles = True


model_name = "Moments model"
# Paths to be set by user:
model = "moments_full"
non_jeffrey_priors = "modified_azzoni_priors"
data_path_1 = f"/mnt/users/eastonf/Jeffreys/BBPower/chains/all_channels/gaussian_fgs/{model}/jeffreys_priors"
data_path_2 = f"/mnt/users/eastonf/Jeffreys/BBPower/chains/all_channels/gaussian_fgs/{model}/{non_jeffrey_priors}"
out_path = f"/mnt/users/eastonf/Jeffreys/BBPower/comparison_plots/{model}/{non_jeffrey_priors}/real_val_is_fiducial_mean"



### Dictionaries describing the parameters ###

# Holds the 'real' values of all parameters
REAL_VALS = {
    "r_tensor": 0,
    "A_lens": 1,
    "amp_s_bb": 1.6,
    "amp_d_bb": 28,
    "alpha_s_bb":-1,
    "alpha_d_bb":-0.16,
    "beta_s":-3,
    "beta_d": 1.54,
    "epsilon_ds":0,
    "amp_d_beta": None,
    "amp_s_beta": None,
    "gamma_d_beta":None,
    "gamma_s_beta":None
}

# Holds the mean of the posterior means for the fiducial model using Jeffreys priors
REAL_VALS = {
    "r_tensor": 0.001267832030207712,
    "A_lens": 0.945418990656891,
    "amp_s_bb": 1.5917463576485205,
    "amp_d_bb": 27.92300381254014,
    "alpha_s_bb":-0.9298019230744116,
    "alpha_d_bb":-0.21402735615145801,
    "beta_s":-3.0118617944021673,
    "beta_d": 1.5410212490255148,
    "epsilon_ds":0.0010098745815456311,
    "amp_d_beta": None,
    "amp_s_beta": None,
    "gamma_d_beta":None,
    "gamma_s_beta":None
}

# Holds the labels for parameters (in latex)
LATEX_NAMES = {
    "r_tensor": r"$r_{tensor}$",
    "A_lens": r"$A_{lens}$",
    "amp_s_bb": r"$A_s^{BB}$",
    "amp_d_bb": r"$A_d^{BB}$",
    "alpha_s_bb":r"$\alpha_s^{BB}$",
    "alpha_d_bb":r"$\alpha_d^{BB}$",
    "beta_s":r"$\beta_s$",
    "beta_d": r"$\beta_d$",
    "epsilon_ds":r"$\epsilon_{ds}$",
    "amp_s_beta":r"$B_s$",
    "amp_d_beta":r"$B_d$",
    "gamma_s_beta":r"$\gamma_s$",
    "gamma_d_beta":r"$\gamma_d$"
}




def make_plots(axes, title, axis_label, data, val):
    make_hist(axes[0],f"{title}",axis_label,data[0],data[1],val)
    make_box_plot(axes[1],f"{title}",axis_label,data[0],data[1],val)

def make_hist(ax, title, xlabel, data1, data2, val):

    # Plot values and add axis
    #ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title)

    # Plot data    
    minimum = min(data1+data2)
    maximum = max(data1+data2)
    ax.hist(data1,bins=10,facecolor="gold", range=(minimum,maximum), label = "Jeffreys prior")
    ax.hist(data2,bins=10,facecolor="cyan",alpha=0.45, range=(minimum,maximum), label = "Non-Jeffreys prior")
    if data1:
        mean1 = np.mean(data1)
        ax.axvline(mean1, c='darkgoldenrod', ls = ':')
    if data2:
        mean2 = np.mean(data2)
        ax.axvline(mean2, c='darkcyan', ls = ':')

    if val is not None:
        ax.axvline(val, c='dimgrey',ls='-',label = "'Real' Value")
        
        if xlabel not in [r"$A_s^{BB}$",r"$A_d^{BB}$"]:
            lower,upper = ax.get_xlim()
            lower -= val
            upper -= val
            half_width = max(abs(lower), upper)
            ax.set_xlim(xmin=val-half_width, xmax=val + half_width)

    ax.legend()
    fig.tight_layout()

def make_box_plot(ax, title, ylabel, data1, data2, val):
    labels = ['Jeffreys Prior', 'Non-Jeffreys Prior']
    colors = ['gold', 'cyan']

    #ax.set_title(title)
    ax.set_xlabel(ylabel, fontsize=12)

    bplot = ax.boxplot([data1,data2],
                       patch_artist=True,  # fill with color
                       tick_labels=labels,  # will be used to label x-ticks
                       orientation = "horizontal",
                       widths=0.4)
    ax.set_yticks([])

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    if val is not None:
        ax.axvline(val,c="dimgrey", label = "'Real' value")
    #ax.legend()

if plot_hists == True:
    
    print("\nReading data for histograms and boxplots")

    if not os.path.exists(f"{out_path}/hist_and_box_plots"):
        os.makedirs(f"{out_path}/hist_and_box_plots")

    data_paths = [data_path_1, data_path_2]

    # Determine the list of all free parameters
    npz = f"{data_path_2}/0010/chi2.npz"
    chain = np.load(npz)
    names = chain["names"]
    nparams = len(names)

    is_first_itter = True
    for param in names:
        val = REAL_VALS[param]
        param_latex = LATEX_NAMES[param]

        mpe = [[] for _ in range(2)]
        mmpe = [[] for _ in range(2)]
        mean = [[] for _ in range(2)]
        median = [[] for _ in range(2)]
        probability = [[] for _ in range(2)]
        reduced_chi2 = [[] for _ in range(2)]

        # Loop through the different priors
        for i in range(2):

            # Loop through the simulations
            for sim_id in range(0,101):

                # Load data files
                chi2_path = f"{data_paths[i]}/{sim_id:04}/chi2.npz"
                emcee_path = f"{data_paths[i]}/{sim_id:04}/emcee.npz"

                try:
                    chi2 = np.load(chi2_path)
                except:
                    if is_first_itter:
                        print(f"WARNING: {data_paths[i]}/{sim_id:04} has no chi2 file")
                    continue

                try: 
                    emcee = np.load(emcee_path)
                except:
                    if is_first_itter:
                        print(f"WARNING: {data_paths[i]}/{sim_id:04} has no emcee file")
                    continue

                # Get posterior maxima
                chi2_names = chi2["names"]
                posterior_max = chi2["params"]
                idx = np.where(chi2_names == param)
                idx = idx[0][0]
                mpe[i].append(posterior_max[idx])
                
                # Get chi2s
                reduced_chi2[i].append(chi2["chi2"] / (chi2["ndof"]-nparams))

                # Get probability parameter < val
                chain = emcee["chain"]
                emcee_names = emcee["names"]
                idx = np.where(emcee_names == param)
                idx = idx[0][0]
                chain_vals = [[chain[i][j][idx] for j in range(len(chain[i]))] for i in range(len(chain))]

                # Get medians
                median[i].append(np.median(chain_vals))

                # Get means
                mean[i].append(np.mean(chain_vals))

                # Get maximum marginalised posterior estimate
                mmpe[i].append(np.mean(chain_vals))

                # Find the probability (using the posterior) that the parameter is greater than the 'real' value
                if val is not None:
                    x = [[chain[i][j][idx] > val for j in range(len(chain[i]))] for i in range(len(chain))]
                    probability[i].append(np.sum(x)/np.size(x))

    
        # Plot graphs
        if is_first_itter:
            print("\nPlotting histograms and boxplots")
            fig, axes = plt.subplots(2,1, figsize=[5,8], gridspec_kw={'height_ratios': [2,1]}, sharex="col")
            make_plots([axes[0],axes[1]], r"Reduced $\chi^2$", r"Reduced $\chi^2$", reduced_chi2, 1)
            fig.suptitle(model_name, fontsize=14)
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            fig.savefig(f"{out_path}/hist_and_box_plots/reduced_chi2.png")
            print(f"{out_path}/hist_and_box_plots/reduced_chi2.png")
            
        if val is not None:
            fig, axes = plt.subplots(2,4, figsize=[20,8], gridspec_kw={'height_ratios': [2,1]}, sharex="col")
            make_plots([axes[0][3],axes[1][3]],f"Probability of Exceding", "Probability", probability,0.5)
        else: 
            fig,axes = plt.subplots(2,3, figsize=[15,8], gridspec_kw={'height_ratios': [2,1]}, sharex="col")
        make_plots([axes[0][0],axes[1][0]], "Posterior Maximum", param_latex, mpe, val)
        make_plots([axes[0][1],axes[1][1]], "Mean", param_latex, mean, val)
        make_plots([axes[0][2],axes[1][2]], "Median", param_latex,median, val)
        fig.suptitle(f"{model_name} - {param_latex} summary statistics", fontsize=14)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.savefig(f"{out_path}/hist_and_box_plots/{param}.png")

        print(f"{out_path}/hist_and_box_plots/{param}.png")
        is_first_itter = False

# Sets the color scheme for the graphs
if plot_triangles == True:

    for sim_id in [24]:

        print(f"\nPlotting triangle plot for {sim_id:04}")


        # output location
        output_fname = f"{out_path}/triangles/{sim_id:04}.png"

        # what parameters to plot
        params_plot = ["r_tensor", "A_lens", "beta_d", "amp_d_bb", "alpha_d_bb", "amp_d_beta","gamma_d_beta","beta_s", "amp_s_bb", "alpha_s_bb", "amp_s_beta", "gamma_s_beta","epsilon_ds"]

        colors = ["darkorange", "navy"]



        # Try to load the emcee files. If a file is missing, skips the current simulation and moves on to the next
        chains = {
            "Jeffreys Prior": f"{data_path_1}/{sim_id:04}/emcee.npz",
            "Non-Jeffreys Prior": f"{data_path_2}/{sim_id:04}/emcee.npz"
        }
        missing_file = False
        for key in chains.keys():
            try:
                chains[key] = np.load(chains[key])
            except:
                print("WARNING: No emcee file for {data_path_1}/{sim_id:04} found. Cannot plot traingle plot for {sim_id:04}")
                missing_file = True
                continue
        if missing_file:
            continue
        
        samples = []
        for case, chain in chains.items():

            # Select only selected parameters for which we have labels
            names_common = list(set(list(chain['names']))
                                & LATEX_NAMES.keys()
                                & set(params_plot))
            msk_common = np.array([n in names_common
                                   for n in chain['names']])
            _, nsamp, npar_chain = chain['chain'].shape
            chain_arr = chain['chain'][:, nsamp//4:, :].reshape([-1, npar_chain])[:, msk_common]  # noqa
            names_common = np.array(chain['names'])[msk_common]
            labels = [LATEX_NAMES[n].replace("$", "") for n in names_common]

            # Getdist
            samples += [
                MCSamples(
                    samples=chain_arr,
                    names=names_common,
                    labels=labels,
                    label=case,
                )
            ]
        g = gplots.getSubplotPlotter()
        g.settings.title_limit_fontsize = 1
        g.settings.axes_fontsize = 14
        g.settings.axes_labelsize = 18
        g.settings.legend_fontsize = 18
        g.triangle_plot(
            samples,
            title_limit = [],
            filled=True,
            legend_loc = 'upper right',
            contour_colors=colors[:len(chains)],
            markers = REAL_VALS,
            marker_args = {"c":"r", "ls":"-"},
            legend_fontsize = 18,
        )
        # Save
        g.export(output_fname)
        print(output_fname)

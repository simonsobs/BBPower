import numpy as np
from utils import (Bpass, get_nmt_binning, get_component_spectra,
                   get_convolved_seds, read_beam_window)
import noise_calc as nc
import sacc
import sys
import os
import yaml
import argparse


def generate_SO_spectra(args):
    """
    Compute theory CMB, synchrotron and dust power spectra from fiducial
    template, beam- and bandpass-convolves them, and saves the multifrequency
    power spectra inside a SACC file.
    """
    # Load the configuration file
    print("  Loading configuration file")
    fname_config = args.globals

    with open(fname_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    nside = config["global"]["nside"]
    delta_ell = config["global"]["delta_ell"]
    band_names = list(config["global"]["map_sets"].keys())
    beam_files = list(config["global"]["map_sets"][b]["beam_file"]
                      for b in band_names)
    bandpass_files = list(config["global"]["map_sets"][b]["bandpass_file"]
                          for b in band_names)

    theory_params = config["theory_params"]
    cmb_templates = config["theory_params"]["CMB"]["cmb_templates"]

    output_dir = args.outdir

    # Bandpasses
    print("  Loading bandpasses")
    bpss = {n: Bpass(n, b) for n, b in zip(band_names, bandpass_files)}

    # Bandpowers
    print("  Computing bandpowers")
    nmt_binning = get_nmt_binning(nside, delta_ell)
    lmax = nmt_binning.lmax
    ls = np.arange(lmax + 1)
    cl2dl = ls*(ls + 1)/(2*np.pi)
    dl2cl = np.zeros_like(cl2dl)
    dl2cl[1:] = 1/(cl2dl[1:])
    lb = nmt_binning.get_effective_ells()
    n_bins = len(lb)
    lwin = np.zeros((len(ls), n_bins))

    for id_bin in range(n_bins):
        weights = np.array(nmt_binning.get_weight_list(id_bin))
        multipoles = np.array(nmt_binning.get_ell_list(id_bin))
        for il, l in enumerate(multipoles):
            lwin[l, id_bin] = weights[il]

    s_wins = sacc.BandpowerWindow(ls, lwin)

    # lbands = np.linspace(2, lmax, n_bins+1, dtype=int)
    # windows = np.zeros([n_bins, lmax+1])
    # for b, (l0, lf) in enumerate(zip(lbands[:-1], lbands[1:])):
    #     windows[b, l0:lf] = (ls * (ls + 1)/(2*np.pi))[l0:lf]
    #     windows[b, :] /= delta_ell

    # Beams
    print("  Loading beams")
    beams = {n: read_beam_window(b, lmax)
             for n, b in zip(band_names, beam_files)}

    print("  Calculating power spectra")
    # Component spectra
    dls_comp = np.zeros([3, 2, 3, 2, lmax + 1]) #[ncomp,np,ncomp,np,nl]
    (dls_comp[1, 0, 1, 0, :],
     dls_comp[1, 1, 1, 1, :],
     dls_comp[2, 0, 2, 0, :],
     dls_comp[2, 1, 2, 1, :],
     dls_comp[0, 0, 0, 0, :],
     dls_comp[0, 1, 0, 1, :]) = get_component_spectra(
         lmax, config["theory_params"], fname_camb_lens_nobb=cmb_templates[0]
    )
    dls_comp *= dl2cl[None, None, None, None, :]

    # Convolve with bandpower windows
    #bpw_comp=np.sum(dls_comp[:,:,:,:,None,:]*windows[None,None,None,None,:,:],axis=5)
    bpw_comp = np.sum(
        dls_comp[:, :, :, :, None, :] * lwin.T[None, None, None, None, :, :],
        axis=5
    )

    # Convolve with bandpasses
    seds = get_convolved_seds(band_names, bpss, theory_params)
    _, nfreqs = seds.shape

    # Components -> frequencies
    bpw_freq_sig = np.einsum('ik,jm,iljno', seds, seds, bpw_comp)

    # N_ell
    # TODO: accept general theory power spectra
    # sens = 2
    # knee = 1
    # ylf = 1
    fsky = 0.1
    nell = np.zeros([nfreqs, lmax+1])
    # _, nell[:,2:], _ = nc.Simons_Observatory_V3_SA_noise(sens, knee,
    #                                                      ylf, fsky,
    #                                                      lmax + 1, 1)
    n_bpw = np.sum(nell[:, None, :]*lwin.T[None,:,:], axis=2)
    bpw_freq_noi = np.zeros_like(bpw_freq_sig)
    for ib, n in enumerate(n_bpw):
        bpw_freq_noi[ib, 0, ib, 0, :] = n_bpw[ib, :]
        bpw_freq_noi[ib, 1, ib, 1, :] = n_bpw[ib, :]

    # Add noise to signal
    bpw_freq_tot = bpw_freq_sig + bpw_freq_noi
    bpw_freq_tot = bpw_freq_tot.reshape([nfreqs*2, nfreqs*2, n_bins])
    bpw_freq_sig = bpw_freq_sig.reshape([nfreqs*2, nfreqs*2, n_bins])
    bpw_freq_noi = bpw_freq_noi.reshape([nfreqs*2, nfreqs*2, n_bins])

    # Creating Sacc files
    s_d = sacc.Sacc()
    s_f = sacc.Sacc()
    s_n = sacc.Sacc()

    # Adding tracers
    print("  Adding sacc tracers")
    for ib, n in enumerate(band_names):
        bandpass = bpss[n]
        beam = beams[n]
        for s in [s_d, s_f, s_n]:
            # print(n)
            # print(bandpass.nu)
            # print(bandpass.bnu)
            # print(ls)
            # print(beam)
            s.add_tracer('NuMap', n,
                         quantity='cmb_polarization',
                         spin=2,
                         nu=bandpass.nu,
                         bandpass=bandpass.bnu,
                         ell=ls,
                         beam=beam,
                         nu_unit='GHz',
                         map_unit='uK_CMB')

    # Adding power spectra
    print("  Adding power spectra to sacc")
    nmaps = 2*nfreqs
    ncross = (nmaps*(nmaps + 1))//2
    indices_tr = np.triu_indices(nmaps)
    map_names=[]

    for ib, n in enumerate(band_names):
        map_names.append(n + '_E')
        map_names.append(n + '_B')

    for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
        n1 = map_names[i1][:-2]
        n2 = map_names[i2][:-2]
        p1 = map_names[i1][-1].lower()
        p2 = map_names[i2][-1].lower()
        cl_type = f'cl_{p1}{p2}'
        s_d.add_ell_cl(cl_type, n1, n2, lb, bpw_freq_sig[i1, i2, :],
                       window=s_wins)
        s_f.add_ell_cl(cl_type, n1, n2, lb, bpw_freq_sig[i1, i2, :],
                       window=s_wins)
        s_n.add_ell_cl(cl_type, n1, n2, lb, bpw_freq_noi[i1, i2, :],
                       window=s_wins)

    # Add covariance
    n_split = 4
    def is_cross(s1,f1,s2,f2):
        """
        Given two maps with split s1 frequence f1 and split s2 and frequency f2
        Determine whether the power spectrum is cross spectrum.
        """
        if s1 != s2:
            return True
        else:
            if f1 != f2:
                return True
            else:
                return False
    def get_cross_splits(f1,f2):
        """
        Given two frequency f1 and f2, get all map splits pairs that the power
        spectrum only contain cross spectrum.
        Return split pairs and total number of cross spectrum.
        """
        if f1 == f2:
            # If freq are the same, then all cross pairs would have different
            # split, so the number of total cross pairs is:
            num_cross = n_split*(n_split-1)/2
            split_pairs = np.triu_indices(n_split,1)
            assert num_cross == split_pairs[0].size
            return split_pairs, num_cross
        else:
            # If freq are different, then any split combination will be cross pair,
            # so the number of total cross pairs is:
            num_cross = n_split**2
            split_pairs = np.triu_indices(n_split,-n_split)
            assert num_cross == split_pairs[0].size
            return split_pairs, num_cross
    print("  Adding covariance to sacc")
    cov_bpw = np.zeros([ncross, n_bins, ncross, n_bins])
    factor_modecount = 1./((2*lb + 1)*delta_ell*fsky)
    for ii, (m1, m2) in enumerate(zip(indices_tr[0], indices_tr[1])):
        for jj, (m3, m4) in enumerate(zip(indices_tr[0], indices_tr[1])):
            # frequency index of each map
            f1 = m1//2
            f2 = m2//2
            f3 = m3//2
            f4 = m4//2
            split_pairs_12, num_cross_12 = get_cross_splits(f1,f2)
            split_pairs_34, num_cross_34 = get_cross_splits(f3,f4)
            for s_alpha, s_beta in zip(*split_pairs_12):
                for s_mu, s_nu in zip(*split_pairs_34):
                    if is_cross(s_alpha, f1, s_mu, f3):
                        Cell_13 = bpw_freq_sig[m1,m3,:]
                    else:
                        Cell_13 = bpw_freq_tot[m1,m3,:]

                    if is_cross(s_beta, f2, s_nu, f4):
                        Cell_24 = bpw_freq_sig[m2,m4,:]
                    else:
                        Cell_24 = bpw_freq_tot[m2,m4,:]

                    if is_cross(s_alpha, f1, s_nu, f4):
                        Cell_14 = bpw_freq_sig[m1,m4,:]
                    else:
                        Cell_14 = bpw_freq_tot[m1,m4,:]

                    if is_cross(s_beta, f2, s_mu, f3):
                        Cell_23 = bpw_freq_sig[m2,m3,:]
                    else:
                        Cell_23 = bpw_freq_tot[m2,m3,:]

                    cov_bpw[ii,:,jj,:] += np.diag(
                            factor_modecount
                            * (1/num_cross_12)
                            * (1/num_cross_34)
                            * ( Cell_13 * Cell_24 + Cell_14 * Cell_23)
                            )
    cov_bpw = cov_bpw.reshape([ncross * n_bins, ncross * n_bins])
    s_d.add_covariance(cov_bpw)

    # Write output
    print("  Writing sacc files")
    s_d.save_fits(f"{output_dir}/cls_coadd.fits", overwrite=True)
    s_f.save_fits(f"{output_dir}/cls_fid.fits", overwrite=True)
    s_n.save_fits(f"{output_dir}/cls_noise.fits", overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing external data for the pipeline")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--outdir", type=str,
                        help="Path to the output directory")

    args = parser.parse_args()

    generate_SO_spectra(args)

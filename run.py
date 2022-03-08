#!/usr/bin/env python3

#############################################################################################
#############################################################################################
# Refactored and edited by Kylie Ying. Original code developed by Jasmine Brewer and Andrew #
# Turner with conceptual oversight from Jesse Thaler                                        #
# Original code located at: https://github.com/jasminebrewer/jet-topics-from-MCMC           #
# Last modified Mar-08-2022                                                                 #
#############################################################################################
#############################################################################################

#### A valuable tutorial on fitting with MCMC: https://emcee.readthedocs.io/en/stable/tutorials/line/ ####

import argparse
import math
import csv
import numpy as np
import os
import errno
import pickle as pkl

import config
from data import Data
from plotter import Plotter
from model import Model


def run(system, sample1, sample2, nwalkers, nsamples, burn_in, nkappa, min_bin, max_bin, min_pt, max_pt, xlabel):
    ###################################################################################
    ## 1. Initialize all the objects (for data, plotting, and modeling)              ##
    ###################################################################################
    # format all the data, check out data.py to see what's actually going on
    data_obj = Data(sample1_label=sample1, sample2_label=sample2,
                    data_file_path=f"{config.DATA_FOLDER_PATH}/{system}.csv", min_bin=min_bin, max_bin=max_bin, min_pt=min_pt, max_pt=max_pt)

    # we use data_obj's min/max bins because data_obj refines the bins from the input
    min_bin, max_bin = data_obj.min_bin, data_obj.max_bin

    plotter = Plotter(data_obj=data_obj, system=system, sample1=sample1, sample2=sample2, nwalkers=nwalkers, nsamples=nsamples,
                      burn_in=burn_in, xlabel=xlabel, min_bin=min_bin, max_bin=max_bin)
    plotter.plot_unnormalized_input()  # plot the unnormalized input

    model = Model(data_obj=data_obj, system=system, nwalkers=nwalkers, nsamples=nsamples,
                  burn_in=burn_in, nkappa=nkappa, min_bin=min_bin, max_bin=max_bin)

    print("Initialized objects")

    print(f"Photonjet: {data_obj.sample1.tot_n}, gluon: {data_obj.photon_gluons.tot_n}, quark: {data_obj.photon_quarks.tot_n}")
    print(f"Dijet: {data_obj.sample2.tot_n}, gluon: {data_obj.dijet_gluons.tot_n}, quark: {data_obj.dijet_quarks.tot_n}")

    ###################################################################################
    ## 2. Get least squares fit for starting point of parameter fits                 ##
    ###################################################################################
    # performing least squares fit
    lst_sq_fit = model.get_least_squares_fit(
        trytimes=config.LEAST_SQUARES_TRY_TIMES, initial_point=config.FIT_INITIAL_POINT, perturb=config.LEAST_SQUARES_PERTURB)
    plotter.plot_least_squares(lst_sq_fit)  # plotting results for least squares

    print("Finished least squares fit")

    ###################################################################################
    ## 3. Collect MCMC samples                                                       ##
    ###################################################################################
    samples = model.get_MCMC_samples(lst_sq_fit, variation_factor=config.MCMC_PERTURB)
    plotter.plot_mcmc_samples(samples)  # plotting mcmc samples

    print("Finished MCMC")

    ###################################################################################
    ## 4. Extract kappas using MCMC samples                                          ##
    ###################################################################################
    kappas_ab_arg, kappas_ab, kappas_ba_arg, kappas_ba, bins_ab, ratios_ab, bins_ba, ratios_ba = model.get_kappas(
        samples)
    del samples  # delete this to limit space
    plotter.plot_kappas(kappas_ab_arg, kappas_ab, kappas_ba_arg, kappas_ba, bins_ab, ratios_ab, bins_ba, ratios_ba)

    # write kappas to file
    try:
        os.makedirs("kappas")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(f'kappas/{system}_{sample1}_{sample2}_{plotter.file_prefix}.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerows([list(kappas) for kappas in [kappas_ab, kappas_ba]])

    # ######### Alternatively, you can set the kappas if you want #####################

    # truth values.. try these!
    # kappas_ab = [0.12689456105002625]
    # kappas_ba = [0.4478009095245291]

    # You can uncomment the following code and start the script here if you want to just start from a
    # saved kappa file instead of the MCMC (you still need to initialize objects in step 1)
    # with open(f'kappas/{system}_{sample1}_{sample2}_{plotter.file_prefix}.csv', 'r') as f:
    #     csv_reader = list(csv.reader(f))
    #     kappas_ab = [float(i) for i in csv_reader[0]]
    #     kappas_ba = [float(i) for i in csv_reader[1]]

    ###################################################################################
    ## 5. Extract fractions and topics                                               ##
    ###################################################################################
    # fa, fb, fa_std, fb_std = model.calc_fracs_from_kappa(kappas_ab, kappas_ba)  # you can calculate but currently not used

    topic1, topic1_err, topic2, topic2_err = model.calc_topics(kappas_ab, kappas_ba)
    plotter.plot_topics(topic1, topic1_err, topic2, topic2_err, color1=config.COLOR1, color2=config.COLOR2)

    ###################################################################################
    ## 6. Extract substructure observables                                           ##
    ###################################################################################

    # kappas
    kappa_ab_mean, kappa_ab_std = np.mean(kappas_ab), np.std(kappas_ab)
    kappa_ba_mean, kappa_ba_std = np.mean(kappas_ba), np.std(kappas_ba)

    # save substructure observables
    substructures = {}

    for substructure in ["jet-shape", "jet-frag", "jet-mass", "jet-splitting"]:
        x, quark_vals, gluon_vals, _, _, topic1_vals, topic2_vals = model.calc_substructure(
            substructure, kappa_ab_mean, kappa_ba_mean, kappa_ab_std, kappa_ba_std)
        plotter.plot_substructure(substructure, x, quark_vals, gluon_vals, topic1_vals,
                                  topic2_vals, color1=config.COLOR1, color2=config.COLOR2)
        substructures[substructure] = {"x": x, "quark": quark_vals, "gluon": gluon_vals, "topic1": topic1_vals, "topic2": topic2_vals}

    # saving substructure dict as pickle file so we can use it easily for the modification plots
    # NOTE: now, to get the modification plots, run modification.py
    with open(f"./substructure/output/{system}_{sample1}_{sample2}_{plotter.file_prefix}.pkl", "wb+") as f:
        pkl.dump(substructures, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # this should be string representing the csv file name, if csv file is ./data/150_pt100_qgfracs.csv, user should input 150_pt100_qgfracs
    parser.add_argument('system')
    parser.add_argument('sample1')
    parser.add_argument('sample2')
    parser.add_argument('nwalkers', type=int)
    parser.add_argument('nsamples', type=int)
    parser.add_argument('burn_in', type=int)
    parser.add_argument('nkappa', type=int)
    parser.add_argument('min_bin', type=int, default=-math.inf)
    parser.add_argument('max_bin', type=int, default=math.inf)
    parser.add_argument('min_pt', type=int, default=80)
    parser.add_argument('max_pt', type=int, default=100)
    parser.add_argument('xlabel', default="Constituent Multiplicity", nargs='?')
    args = parser.parse_args()

    run(args.system, args.sample1, args.sample2, args.nwalkers, args.nsamples,
        args.burn_in, args.nkappa, args.min_bin, args.max_bin, args.min_pt, args.max_pt, args.xlabel)

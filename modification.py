#!/usr/bin/env python3

#############################################################################################
#############################################################################################
# Code developed by Kylie Ying with conceptual oversight from Jasmine Brewer, Yi Chen, and  #
# Yen-Jie Lee                                                                               #
# Last modified Mar-08-2022                                                                 #
#############################################################################################
#############################################################################################

import argparse
import pickle as pkl
import config
import os
import numpy as np
import errno
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from plotter import make_error_boxes

# helper functions for jet mass plotting
####################################################################################
def combine_bins_helper(bins, vals, vals_err, t="x"):
    if t == "x":
        v = (bins[0]+bins[-1]+1) / 2
        v_err = v - bins[0]
    elif t == "y":
        v = sum([vals[i] for i in bins])
        v_err = sum([vals_err[i]**2 for i in bins])**0.5
    return v, v_err

def combine_bins(list_of_bins, vals, vals_err, t="x"):
    new_len = len(vals) - sum([len(b) for b in list_of_bins]) + len(list_of_bins)
    bin_starts = [b[0] for b in list_of_bins]
    new_vals, new_vals_err = np.zeros(new_len), np.zeros(new_len)
    i = 0
    new_i = 0
    while new_i < new_len:
        if i in bin_starts:
            bins = list_of_bins[bin_starts.index(i)]
            v, v_err = combine_bins_helper(bins, vals, vals_err, t)
            new_vals[new_i] = v
            new_vals_err[new_i] = v_err
            i = bins[-1] + 1
        else:
            new_vals[new_i] = vals[i]
            new_vals_err[new_i] = vals_err[i]
            i += 1
            
        new_i += 1
        
    return new_vals, new_vals_err
####################################################################################

def get_substructures(system, sample1, sample2, file_prefix):
    with open(f"./substructure/output/{system}_{sample1}_{sample2}_{file_prefix}.pkl", "rb") as f:
        return pkl.load(f)

def get_ratio_and_error(hi, pp):
    y = hi[:,0] / pp[:,0]
    relative_err = np.sqrt(np.square(hi[:,1] / hi[:,0]) + np.square(pp[:,1] / pp[:,0]))
    return y, relative_err * y

def plot_substructure(substructure, pp_substructures, hi_substructures, system, file_prefix, color1="salmon", color2="steelblue"):
    folder = f"{config.MOD_PLOT_OUTPUT_PATH}/{system}"
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    pp_calc = pp_substructures[substructure]
    hi_calc = hi_substructures[substructure]

    x = pp_calc["x"][:,0]
    x_err = pp_calc["x"][:,1]

    quark_y, quark_err = get_ratio_and_error(hi_calc["quark"], pp_calc["quark"])
    gluon_y, gluon_err = get_ratio_and_error(hi_calc["gluon"], pp_calc["gluon"])
    topic1_y, topic1_err = get_ratio_and_error(hi_calc["topic1"], pp_calc["topic1"])
    topic2_y, topic2_err = get_ratio_and_error(hi_calc["topic2"], pp_calc["topic2"])

    if config.SUBSTRUCTURES[substructure]["mod_combine_bins"]:
        x, x_err = combine_bins(config.SUBSTRUCTURES[substructure]["mod_combine_bins"], x, x_err, t="x")
        quark_y, quark_err = combine_bins(config.SUBSTRUCTURES[substructure]["mod_combine_bins"], quark_y, quark_err, t="y")
        gluon_y, gluon_err = combine_bins(config.SUBSTRUCTURES[substructure]["mod_combine_bins"], gluon_y, gluon_err, t="y")
        topic1_y, topic1_err = combine_bins(config.SUBSTRUCTURES[substructure]["mod_combine_bins"], topic1_y, topic1_err, t="y")
        topic2_y, topic2_err = combine_bins(config.SUBSTRUCTURES[substructure]["mod_combine_bins"], topic2_y, topic2_err, t="y")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, quark_y, yerr=None, xerr=x_err, fmt='o', color='k', label="Quark")
    ax.errorbar(x, gluon_y, yerr=None, xerr=x_err, fmt='o', color='dimgray', label="Gluon")
    ax.errorbar(x, topic1_y, yerr=None, xerr=x_err, fmt='o', color=color1, label="Topic 1")
    ax.errorbar(x, topic2_y, yerr=None, xerr=x_err, fmt='o', color=color2, label="Topic 2")

    make_error_boxes(ax, x, quark_y, x_err, quark_err, facecolor='k')
    make_error_boxes(ax, x, gluon_y, x_err, gluon_err, facecolor='dimgray')
    make_error_boxes(ax, x, topic1_y, x_err, topic1_err, facecolor=color1)
    make_error_boxes(ax, x, topic2_y, x_err, topic2_err, facecolor=color2)

    ax.axhline(y=1, color='k', linestyle=':')

    ax.set_ylabel("Ratio (HI / PP)")
    ax.set_xlabel(config.SUBSTRUCTURES[substructure]["xlabel"])

    if config.SUBSTRUCTURES[substructure]["xlim_mod"] : ax.set_xlim(config.SUBSTRUCTURES[substructure]["xlim_mod"])
    if config.SUBSTRUCTURES[substructure]["ylim_mod"] : ax.set_ylim(config.SUBSTRUCTURES[substructure]["ylim_mod"])

    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    plt.title(f"{config.SUBSTRUCTURES[substructure]['title']} Modification")
    plt.legend()
    plt.savefig(f'{folder}/{file_prefix}_{substructure}.png')
    plt.clf()

def run(system, pp_sample1, pp_sample2, hi_sample1, hi_sample2, nwalkers, nsamples, burn_in):
    file_prefix = 'settings_'+str(nwalkers)+','+str(nsamples)+','+str(burn_in)
    
    pp_substructures = get_substructures(system, pp_sample1, pp_sample2, file_prefix)
    hi_substructures = get_substructures(system, hi_sample1, hi_sample2, file_prefix)

    for substructure in ["jet-shape", "jet-frag", "jet-mass", "jet-splitting"]:
        plot_substructure(substructure, pp_substructures, hi_substructures, system, file_prefix, color1="salmon", color2="steelblue")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # this should be string representing the csv file name, if csv file is ./data/150_pt100_qgfracs.csv, user should input 150_pt100_qgfracs
    parser.add_argument('system')
    parser.add_argument('pp_sample1')
    parser.add_argument('pp_sample2')
    parser.add_argument('hi_sample1')
    parser.add_argument('hi_sample2')
    parser.add_argument('nwalkers', type=int)
    parser.add_argument('nsamples', type=int)
    parser.add_argument('burn_in', type=int)
    args = parser.parse_args()

    run(args.system, args.pp_sample1, args.pp_sample2, args.hi_sample1, args.hi_sample2, args.nwalkers, args.nsamples, args.burn_in)
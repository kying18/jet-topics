#!/usr/bin/env python3

import argparse
import math

import config
from data import Histogram, Data
from plotter import Plotter
from multiprocessing import Pool, cpu_count


def run(system, sample1, sample2, nwalkers, nsamples, burn_in, nkappa, min_bin, max_bin, xlabel):
    # format all the data, check out data.py to see what's actually going on
    data_obj = Data(sample1_label=sample1, sample2_label=sample2,
                    data_file_path=f"{config.DATA_FOLDER_PATH}/{system}.csv", min_bin=min_bin, max_bin=max_bin)
    plotter = Plotter(data_obj=data_obj, system=system, nwalkers=nwalkers,
                      nsamples=nsamples, burn_in=burn_in, xlabel=xlabel)

    plotter.plot_unnormalized_input()


if __name__ == '__main__':
    ncpu = cpu_count()
    print(f"{ncpu} CPUs")

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
    parser.add_argument(
        'xlabel', default="Constituent Multiplicity", nargs='?')
    args = parser.parse_args()

    run(args.system, args.sample1, args.sample2, args.nwalkers, args.nsamples,
        args.burn_in, args.nkappa, args.min_bin, args.max_bin, args.xlabel)

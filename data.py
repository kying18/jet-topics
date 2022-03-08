import numpy as np
from collections import namedtuple
import math

# in case you are unfamiliar with namedtuples, I think of this as a simplified object
# you can initialize it by calling Histogram(*args) and the name of the args are space separated in the string below
# I use this as a tidy way to keep track of the histograms
# let's say we have h = Histogram(hist_unnorm, hist_unnorm_error, etc), then we can use h.hist_unnorm to get
# the unnormalized histogram, h.hist_unnorm_error to get unnormalized histogram error, etc.
Histogram = namedtuple(
    "Histogram", "x hist_unnorm hist_unnorm_error hist hist_error tot_n")

# Data object in charge of parsing the data from the data file and creating the histograms that we will use for the MCMC
# Currently, this code caters towards photonjet/dijet inputs and quark/gluon topics


class Data:
    def __init__(self, sample1_label='pp80_photonjet', sample2_label='pp80', data_file_path="./data/pt80100.csv", min_bin=-math.inf, max_bin=math.inf, min_pt=80, max_pt=100):
        self.sample1_label = sample1_label
        self.sample2_label = sample2_label
        self.data_file_path = data_file_path

        self.min_pt = min_pt
        self.max_pt = max_pt

        if sample1_label[:4] == 'pbpb':
            self.sample = 'pbpb80_0_10_wide'
            self.sample_type = 'pbpb'
        else:
            self.sample = sample1_label.split("_")[0]
            self.sample_type = 'pp'

        samples = self.get_data()

        # takes the leftmost nonzero bin as min_bin, and rightmost nonzero bin as max_bin, unless user defines a larger min_bin or smaller max_bin
        # self.min_bin = min_bin
        # self.max_bin = max_bin
        self.min_bin = max(min(np.min(np.nonzero(samples["sample1"])),
                           np.min(np.nonzero(samples["sample2"]))), min_bin)
        self.max_bin = min(max(np.max(np.nonzero(samples["sample1"])),
                           np.max(np.nonzero(samples["sample2"]))), max_bin)

        self.sample1, self.sample2, self.photon_quarks, self.photon_gluons, self.dijet_quarks, self.dijet_gluons = self.format_samples(
            samples)

    @staticmethod
    def get_mean(mylist):
        return np.array([(mylist[i]+mylist[i+1])/2 for i in range(0, len(mylist)-1)])

    @staticmethod
    # in case you are unfamiliar with static methods, these behave like plain functions except that you can call them from an instance or the class
    def make_numpy(array, reverse):
        if reverse:
            return np.array(list(map(lambda x: float(x), array))[::-1])
        else:
            return np.array(list(map(lambda x: float(x), array)))

    def get_data(self, reverse=False):
        # input_filename is the path to the csv file
        # sample1_label/sample2_label is string that represents that we want to use (ie this is the label on the LHS of the csv, like pbpb150_0_10_wide_zjet)
        # sample1_label should probably be different than sample2_label if you want to run this code on multiple samples lol
        samples = {}
        bins = []

        # opening csv file and reading it
        with open(self.data_file_path, 'rt') as f:
            all_lines = f.readlines()

            # parse each individual line of csv file
            for i in range(len(all_lines)):
                split_vals = all_lines[i].split(",")
                for suffix in ["", "_error"]:
                    if split_vals[0] == f'{self.sample1_label}{suffix}':
                        # samples[f'sample1{suffix}'] = np.array(list(map(lambda x: float(x), split_vals[1:-1])))
                        samples[f'sample1{suffix}'] = self.make_numpy(
                            split_vals[1:-1], reverse=reverse)
                    elif split_vals[0] == f'{self.sample2_label}{suffix}':
                        # samples[f'sample2{suffix}'] = np.array(list(map(lambda x: float(x), split_vals[1:-1])))
                        samples[f'sample2{suffix}'] = self.make_numpy(
                            split_vals[1:-1], reverse=reverse)

                for qg_suffix in ["_quark_truth", "_quark_truth_error", "_gluon_truth", "_gluon_truth_error"]:
                    if split_vals[0] == f'{self.sample}_photonjet{qg_suffix}':
                        samples[f'photon{qg_suffix}'] = self.make_numpy(
                            split_vals[1:-1], reverse=reverse)
                    elif split_vals[0] == f'{self.sample}{qg_suffix}':
                        samples[f'dijet{qg_suffix}'] = self.make_numpy(
                            split_vals[1:-1], reverse=reverse)

        # should be a dictionary {'sample1': [histogram values], 'sample1_error': [histogram of errors], ...}
        return samples

    def format_hist(self, hist, hist_error):
        # hist is list of integers rep histogram bins
        hist = hist[self.min_bin:self.max_bin]
        hist_error = hist_error[self.min_bin:self.max_bin]
        tot_n = sum(hist)
        hist_norm = 1/tot_n * hist
        hist_norm_error = 1/tot_n * hist_error
        bins_center = self.get_mean(range(self.min_bin, self.max_bin+1))

        return Histogram(bins_center, hist, hist_error, hist_norm, hist_norm_error, tot_n)

    def format_samples(self, samples):
        # takes the dictionary of samples and creates normalized histogram: [[normalized histogram, normalized histogram errors, histogram of counts], total count]
        # min_bin and max_bin are indices of minimum bin and maximum bin: [min_bin, max_bin)
        # returns this list of histograms/count for sample1, sample2, combined quark, combined gluon (in that order)
        sample1 = self.format_hist(samples['sample1'], samples['sample1_error'])
        sample2 = self.format_hist(samples['sample2'], samples['sample2_error'])
        photon_quarks = self.format_hist(samples['photon_quark_truth'], samples['photon_quark_truth_error'])
        photon_gluons = self.format_hist(samples['photon_gluon_truth'], samples['photon_gluon_truth_error'])
        dijet_quarks = self.format_hist(samples['dijet_quark_truth'], samples['dijet_quark_truth_error'])
        dijet_gluons = self.format_hist(samples['dijet_gluon_truth'], samples['dijet_gluon_truth_error'])

        return sample1, sample2, photon_quarks, photon_gluons, dijet_quarks, dijet_gluons

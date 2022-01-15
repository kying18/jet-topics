import matplotlib.pyplot as plt
import numpy as np
import errno
import os

import config
from model import Model

# Adjusting font sizes for matplotlib settings
# controls default text sizes
plt.rc('font', size=config.SMALL_FONT_SIZE)
# fontsize of the axes title
plt.rc('axes', titlesize=config.LARGE_FONT_SIZE)
# fontsize of the x and y labels
plt.rc('axes', labelsize=config.MEDIUM_FONT_SIZE)
# fontsize of the tick labels
plt.rc('xtick', labelsize=config.SMALL_FONT_SIZE)
# fontsize of the tick labels
plt.rc('ytick', labelsize=config.SMALL_FONT_SIZE)
# legend fontsize
plt.rc('legend', fontsize=config.SMALL_FONT_SIZE)
# fontsize of the figure title
plt.rc('figure', titlesize=config.LARGE_FONT_SIZE)


class Plotter:
    def __init__(self, data_obj, system, nwalkers, nsamples, burn_in, xlabel, min_bin, max_bin):
        # data_obj should be Data object from data.py

        # making folders for results
        try:
            os.makedirs(f'{config.PLOT_OUTPUT_PATH}/{system}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.burn_in = burn_in
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.xlabel = xlabel
        self.system = system
        self.file_prefix = 'settings_'+str(nwalkers)+','+str(nsamples)+','+str(burn_in) + \
            '_' + str(self.min_bin) + '_' + str(self.max_bin)
        self.data_obj = data_obj

        self.bins_center = self.get_mean(range(self.min_bin, self.max_bin+1))
        self.bins = range(self.min_bin, self.max_bin+1)

    @staticmethod
    def get_mean(mylist):
        return np.array([(mylist[i]+mylist[i+1])/2 for i in range(0, len(mylist)-1)])

    def plot_unnormalized_input(self):
        _, ax = plt.subplots(figsize=(8, 6))

        ax.step(self.bins_center, self.data_obj.sample1.hist_unnorm, where='mid', label="Input 1", color='k')
        ax.fill_between(self.bins[1:], self.data_obj.sample1.hist_unnorm-self.data_obj.sample1.hist_unnorm_error,
                        self.data_obj.sample1.hist_unnorm+self.data_obj.sample1.hist_unnorm_error, step='pre', color='k', alpha=0.3)
        ax.step(self.bins_center, self.data_obj.sample2.hist_unnorm, where='mid', label="Input 2", color='gray')
        ax.fill_between(self.bins[1:], self.data_obj.sample2.hist_unnorm-self.data_obj.sample2.hist_unnorm_error,
                        self.data_obj.sample2.hist_unnorm+self.data_obj.sample2.hist_unnorm_error, step='pre', color='gray', alpha=0.3)

        plt.xlabel(self.xlabel)
        plt.ylabel('N')
        plt.legend()
        plt.xlim((self.min_bin, self.max_bin))
        plt.title('Input Histograms (Unnormalized)')
        # this is called an fstring, it replaces the {var} with the value of `var`
        plt.savefig(f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_input_unnormalized.png')
        plt.clf()

    def plot_least_squares(self, lst_sq_fit):
        params, fracs1, fracs2 = lst_sq_fit[:-6], lst_sq_fit[-6:-3], lst_sq_fit[-3:]
        result1, result2 = np.concatenate((params, fracs1)), np.concatenate((params, fracs2))

        # plot normalized histograms
        _, ax = plt.subplots(figsize=(8, 6))
        ax.step(self.bins_center, self.data_obj.sample1.hist, where='mid', label="Input 1", color='k')
        ax.fill_between(self.bins[1:], self.data_obj.sample1.hist-self.data_obj.sample1.hist_error,
                        self.data_obj.sample1.hist+self.data_obj.sample1.hist_error, step='pre', color='k', alpha=0.3)
        ax.step(self.bins_center, self.data_obj.sample2.hist, where='mid', label="Input 2", color='gray')
        ax.fill_between(self.bins[1:], self.data_obj.sample2.hist-self.data_obj.sample2.hist_error,
                        self.data_obj.sample2.hist+self.data_obj.sample2.hist_error, step='pre', color='gray', alpha=0.3)

        # plot least squares fit
        ax.plot(self.bins_center, [Model.model_func(*result1, x)for x in self.bins_center], 'g--', label='Fit 1')
        ax.plot(self.bins_center, [Model.model_func(*result2, x)for x in self.bins_center], 'm--', label='Fit 2')
        plt.xlabel(self.xlabel)
        plt.ylabel('Probability')
        plt.legend()
        plt.xlim((self.min_bin, self.max_bin))
        plt.title('Input Histograms')
        plt.savefig(f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_lstsq_fit.png')
        plt.clf()

    def plot_mcmc_samples(self, samples, ndim=18):
        fig, axes = plt.subplots(ndim, figsize=(10, 16), sharex=True)
        for i in range(ndim):
            axes[i].plot(range(0, self.nsamples), samples[:, :, i], "k", alpha=0.1)
            axes[i].axvline(x=self.burn_in, color='blue')
        fig.suptitle('MCMC Samples')
        plt.savefig(f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_mcmc_samples.png')
        plt.clf()

    def plot_kappas(self):
        # mask1_zeros = hist1_n>0 # used for plotting only
        # mask2_zeros = hist2_n>0 # used for plotting only
        # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8, 6))
        # ax1.plot(real_bins_for_hist[mask2_zeros],hist1[mask2_zeros]/hist2[mask2_zeros],'b--')
        # ax2.plot(real_bins_for_hist[mask1_zeros],hist2[mask1_zeros]/hist1[mask1_zeros],'b--',label='data')
        pass

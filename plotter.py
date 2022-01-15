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

    def plot_kappas(self, kappas_ab_arg, kappas_ab, kappas_ba_arg, kappas_ba, bins_ab, ratios_ab, bins_ba, ratios_ba):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        mask1, mask2 = self.data_obj.sample1.hist_unnorm > 0, self.data_obj.sample2.hist_unnorm > 0
        ax1.plot(self.bins_center[mask2], self.data_obj.sample1.hist[mask2] /
                 self.data_obj.sample2.hist[mask2], 'k--')
        ax2.plot(self.bins_center[mask1], self.data_obj.sample2.hist[mask1] /
                 self.data_obj.sample1.hist[mask1], 'k--', label='Data')

        for i in range(len(kappas_ab_arg)):
            ax1.plot(kappas_ab_arg[i], kappas_ab[i], 'bo', alpha=0.1)
            ax1.plot(bins_ab, ratios_ab[i], color='r', alpha=0.1)
            if i == 0:
                ax2.plot(kappas_ba_arg[i], kappas_ba[i], 'bo', alpha=0.1, label="Extracted kappas")
                ax2.plot(bins_ba, ratios_ba[i], color='r', alpha=0.1, label="MCMC fits")
            else:
                ax2.plot(kappas_ba_arg[i], kappas_ba[i], 'bo', alpha=0.1)
                ax2.plot(bins_ba, ratios_ba[i], color='r', alpha=0.1)

        ax1.set_ylim((0, 3.5))
        ax2.set_ylim((0, 3.5))
        ax1.set_ylabel('Input A / Input B')
        ax2.set_ylabel('Input B / Input A')
        ax1.set_xlabel(self.xlabel)
        ax2.set_xlabel(self.xlabel)
        ax1.set_title("MCMC Fit and Extracted Kappas")
        ax2.set_title("MCMC Fit and Extracted Kappas")
        ax2.legend()
        fig.suptitle('MCMC Fit and Extracted Kappas')
        plt.savefig(f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_kappas.png')
        plt.clf()

    def plot_topics(self, topic1, topic1_err, topic2, topic2_err, color1="purple", color2="green"):
        # photon_q = self.data_obj.photon_quarks.hist
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.step(self.bins_center, topic1, where='mid', color=color1, label='Topic 1')
        ax.fill_between(self.bins[1:], topic1-topic1_err, topic1+topic1_err, step='pre', color=color1, alpha=0.3)
        ax.step(self.bins_center, topic2, where='mid', color=color2, label='Topic 2')
        ax.fill_between(self.bins[1:], topic2-topic2_err, topic2+topic2_err, step='pre', color=color2, alpha=0.3)

        ax.plot(self.bins_center, self.data_obj.photon_quarks.hist, color='k', label=r'$\gamma$+q')
        ax.plot(self.bins_center, self.data_obj.photon_gluons.hist,
                color='k', linestyle='--', dashes=(6, 2), label=r'$\gamma$+g')

        ax.plot(self.bins_center, self.data_obj.dijet_quarks.hist, color='k', dashes=(3, 1), label='Dijet q')
        ax.plot(self.bins_center, self.data_obj.dijet_gluons.hist,
                color='k', linestyle=':', dashes=(1, 1), label='Dijet g')

        ax.set_xlim((self.min_bin, self.max_bin))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Probability')
        ax.legend()
        plt.title('Resulting Topics')

        plt.savefig(f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_topics.png')
        plt.clf()

import matplotlib.pyplot as plt
import numpy as np
import errno
import os

import config

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
    def __init__(self, data_obj, system, nwalkers, nsamples, burn_in, xlabel):
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
        self.min_bin = data_obj.min_bin
        self.max_bin = data_obj.max_bin
        self.xlabel = xlabel
        self.system = system
        self.file_prefix = 'settings_'+str(nwalkers)+','+str(nsamples)+','+str(
            burn_in) + '_' + str(self.min_bin) + '_' + str(self.max_bin)
        self.data_obj = data_obj

        self.bins_center = self.get_mean(range(self.min_bin, self.max_bin+1))
        self.bins = range(self.min_bin, self.max_bin+1)

    @staticmethod
    def get_mean(mylist):
        return np.array([(mylist[i]+mylist[i+1])/2 for i in range(0, len(mylist)-1)])

    def plot_unnormalized_input(self):
        _, ax = plt.subplots(figsize=(8, 6))
        ax.step(self.bins_center, self.data_obj.sample1.hist,
                where='mid', label="Input 1", color='k')
        ax.fill_between(self.bins[1:], self.data_obj.sample1.hist-self.data_obj.sample1.hist_error,
                        self.data_obj.sample1.hist+self.data_obj.sample1.hist_error, step='pre', color='k', alpha=0.3)
        ax.step(self.bins_center, self.data_obj.sample2.hist,
                where='mid', label="Input 2", color='gray')
        ax.fill_between(self.bins[1:], self.data_obj.sample2.hist-self.data_obj.sample2.hist_error,
                        self.data_obj.sample2.hist+self.data_obj.sample2.hist_error, step='pre', color='gray', alpha=0.3)

        plt.xlabel(self.xlabel)
        plt.ylabel('N')
        plt.legend()
        plt.xlim((self.min_bin, self.max_bin))
        plt.title('Input Histograms (Unnormalized)')
        # this is called an fstring, it replaces the {var} with the value of `var`
        plt.savefig(
            f'{config.PLOT_OUTPUT_PATH}/{self.system}/{self.file_prefix}_input_unnormalized.png')
        plt.clf()

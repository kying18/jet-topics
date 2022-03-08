import csv
import emcee
import math
from multiprocessing import Pool
import numpy as np
import random
from scipy import special
from scipy.optimize import least_squares

import config


class Model:
    def __init__(self, data_obj, system, nwalkers, nsamples, burn_in, nkappa, min_bin, max_bin):
        self.data_obj = data_obj
        self.system = system
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.burn_in = burn_in
        self.nkappa = nkappa
        self.min_bin = min_bin
        self.max_bin = max_bin

    # fitting function for input histograms
    @staticmethod
    def model_func(mu1, omega1, skew1, mu2, omega2, skew2, mu3, omega3, skew3, mu4, omega4, skew4, c1, c2, c3, x):
        # this is the skew gaussian component
        def pdf_skew_gaussian(mu, omega, skew, x):
            return np.exp(-((x-mu)**2)/(2*omega**2))*special.erfc(-(skew*(x-mu))/(np.sqrt(2)*omega))/(omega*np.sqrt(2*np.pi))
        return c1*pdf_skew_gaussian(mu1, omega1, skew1, x) + c2*pdf_skew_gaussian(mu2, omega2, skew2, x) + c3*pdf_skew_gaussian(mu3, omega3, skew3, x) + (1-c1-c2-c3)*pdf_skew_gaussian(mu4, omega4, skew4, x)

    @staticmethod
    def in_bounds(theta):
        params = theta[:-6]
        fractions = np.append(theta[-6:], (1-np.sum(theta[-6:-3]), 1-np.sum(theta[-3:])))

        # parameters must have the specified bounds
        params_in_bounds = [min(config.FIT_BOUNDS[i]) <= params[i] <= max(config.FIT_BOUNDS[i])
                            for i in range(len(params))]

        # fraction parameters must be between 0 and 1
        fracs_in_bounds = [0 <= fractions[i] <= 1 for i in range(len(fractions))]

        return np.all(params_in_bounds) & np.all(fracs_in_bounds)

    ######################
    # because the fit simultaneously describes two histograms with the same parameters but different sets of fractions,
    # you don't know a priori which fractions describe which fit. Identify the fits with a histogram by using the smallest
    # total squared difference between the fits and the histograms
    #####################
    def put_fits_in_order(self, theta, hist1, hist2):
        """
        hist1 and hist2 are Histogram tuples (as defined in data.py)
        """
        def get_square_diff(y1, y2):
            return np.sum((y1-y2)**2)
        params, fracs1, fracs2 = theta[:-6], theta[-6:-3], theta[-3:]

        fitx = np.concatenate((params, fracs1))
        fity = np.concatenate((params, fracs2))

        diff_x1_y2 = get_square_diff([self.model_func(*fitx, x) for x in hist1.x], hist1.hist) + \
            get_square_diff([self.model_func(*fity, x) for x in hist2.x], hist2.hist)
        diff_x2_y1 = get_square_diff([self.model_func(*fitx, x) for x in hist2.x], hist1.hist) + \
            get_square_diff([self.model_func(*fity, x) for x in hist1.x], hist2.hist)

        if diff_x1_y2 < diff_x2_y1:
            return np.concatenate((params, fracs1, fracs2))
        return np.concatenate((params, fracs2, fracs1))

    def get_least_squares_fit(self, trytimes, initial_point, perturb=0.5):
        ################################################################################
        ################ Simultaneous lst sq helper function for fit ###################
        def simultaneous_least_squares(theta, hist1, hist2):
            """
            hist1 and hist2 are Histogram tuples (as defined in data.py)
            """
            # 12 values defining the shared skew normal distributions
            params = theta[:-6]
            # coefficients of SN functions for input 1 (recall the 4th is just 1-sum(c1, c2, c3))
            fracs1 = theta[-6:-3]
            # coefficients of SN functions for input 2
            fracs2 = theta[-3:]

            if self.in_bounds(theta):
                return np.concatenate((self.model_func(*params, *fracs1, hist1.x) - hist1.hist, self.model_func(*params, *fracs2, hist1.x) - hist2.hist))
            return 10e10
        ################################################################################

        # try a least squares fit many times with slightly varying initial points, and keep the best one
        cost = math.inf
        best_fit = None
        for _ in range(0, trytimes):
            new_initial_point = (1+perturb*np.random.randn(len(initial_point)))*initial_point

            fit = least_squares(simultaneous_least_squares, new_initial_point,
                                args=(self.data_obj.sample1, self.data_obj.sample2))

            if fit['cost'] < cost:
                best_fit = fit
                cost = fit['cost']

        best_fit_params = self.put_fits_in_order(
            best_fit['x'], self.data_obj.sample1, self.data_obj.sample2)

        print("Best fit:", best_fit['x'])

        return best_fit_params

    def lnprob_simul(self, theta, hist1, hist2):
        ################################################################################
        ################ Prior, prob, and likelihood functions for MCMC ################
        def lnprior_bnds(theta):
            if self.in_bounds(theta):
                return 0.0
            return -np.inf

        def lnlike_individ(y, yerr, model, n):
            vec = [model[i]-y[i]+y[i]*np.log(y[i])-y[i]*np.log(model[i]) if y[i] >
                   0 else model[i] for i in range(len(y))]
            return -n*np.sum(vec)

        def lnlike_simul(theta, hist1, hist2):
            theta = self.put_fits_in_order(theta, hist1, hist2)
            params, fracs1, fracs2 = theta[:-6], theta[-6:-3], theta[-3:]

            return lnlike_individ(hist1.hist, hist1.hist_error, self.model_func(*params, *fracs1, hist1.x), hist1.tot_n) + \
                lnlike_individ(hist2.hist, hist2.hist_error, self.model_func(*params, *fracs2, hist2.x), hist2.tot_n)
        ################################################################################
        lp = lnprior_bnds(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_simul(theta, hist1, hist2)

    def get_MCMC_samples(self, best_fit_params, variation_factor=1e-2, backend=None):
        pos = []
        while len(pos) < self.nwalkers:
            trial_pos = best_fit_params*(1 + variation_factor*np.random.randn(config.NDIM))
            if self.in_bounds(trial_pos):
                pos.append(trial_pos)

        with Pool(processes=config.N_WORKERS) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, config.NDIM, self.lnprob_simul, args=(
                self.data_obj.sample1, self.data_obj.sample2), pool=pool, backend=backend)
            sampler.run_mcmc(pos, self.nsamples)

        return sampler.get_chain()

    def get_kappa_extraction_domain(self, hist1, hist2):
        decent_stats_idxs = np.where((hist1.hist_unnorm > 30) & (hist2.hist_unnorm > 30))[0]
        left_stats_idx = decent_stats_idxs[:int(len(decent_stats_idxs) / 2)]
        right_stats_idx = decent_stats_idxs[int(len(decent_stats_idxs) / 2):]

        has_left_anchor = []
        for ratio in [hist1.hist/hist2.hist, hist2.hist/hist1.hist]:
            if np.mean(ratio[left_stats_idx]) < np.mean(ratio[right_stats_idx]):
                has_left_anchor.append(True)
            else:
                has_left_anchor.append(False)

        # one ratio must have left anchor and other have right anchor for kappa extraction to work properly
        if sum(has_left_anchor) != 1:  # basically, true = 1 and false = 0, so if sum = 1, then worked
            print('Failed to unambiguously identify right and left anchor points! If the data has low statistics, kappa may not be extracted properly.')

        upsampled_bins = np.arange(self.data_obj.min_bin, self.data_obj.max_bin + 1 /
                                   config.UPSAMPLE_FACTOR, 1/config.UPSAMPLE_FACTOR)  # upsample the x axis
        nonzero_stats_idxs = np.where((hist1.hist_unnorm > 0) | (hist2.hist_unnorm > 0))[0]

        left_cut_0 = np.min(nonzero_stats_idxs) * config.UPSAMPLE_FACTOR
        right_cut_0 = np.max(nonzero_stats_idxs) * config.UPSAMPLE_FACTOR
        left_cut_30 = np.min(decent_stats_idxs) * config.UPSAMPLE_FACTOR
        right_cut_30 = np.max(decent_stats_idxs) * config.UPSAMPLE_FACTOR

        left_bins = upsampled_bins[left_cut_0:right_cut_30+1]
        right_bins = upsampled_bins[left_cut_30:right_cut_0+1]
        if has_left_anchor[0]:  # hist1/hist2 has the left anchor, so we return x range for left then x range for right
            return left_bins, right_bins
        return right_bins, left_bins

    def get_kappas(self, samples):
        # randomly sample "nkappa" points from the posterior on which to extract kappa
        all_index_tuples = [(i, j) for i in range(self.burn_in, len(samples)) for j in range(len(samples[0]))]
        index_tuples = random.sample(all_index_tuples, self.nkappa)
        posterior_samples = np.array([[samples[tup[0], tup[1], i] for i in range(config.NDIM)] for tup in index_tuples])

        kappas_ab, kappas_ba = np.zeros(len(posterior_samples)), np.zeros(len(posterior_samples))
        kappas_ab_arg, kappas_ba_arg = np.zeros(len(posterior_samples)), np.zeros(len(posterior_samples))
        bins_ab, bins_ba = self.get_kappa_extraction_domain(self.data_obj.sample1, self.data_obj.sample2)
        ratios_ab, ratios_ba = [], []

        for i in range(len(posterior_samples)):
            sample = self.put_fits_in_order(posterior_samples[i], self.data_obj.sample1, self.data_obj.sample2)
            params, fracs1, fracs2 = sample[:-6], sample[-6:-3], sample[-3:]

            fit1 = np.concatenate((params, fracs1))
            fit2 = np.concatenate((params, fracs2))

            ratio_ab = [self.model_func(*fit1, x)/self.model_func(*fit2, x) for x in bins_ab]
            ratio_ba = [self.model_func(*fit2, x)/self.model_func(*fit1, x) for x in bins_ba]
            ratios_ab.append(ratio_ab)
            ratios_ba.append(ratio_ba)

            kappa_ab_arg = bins_ab[np.argmin(ratio_ab)]
            kappa_ab = np.min(ratio_ab)

            kappa_ba_arg = bins_ba[np.argmin(ratio_ba)]
            kappa_ba = np.min(ratio_ba)

            # storing kappa values and respective x values
            kappas_ab_arg[i] = kappa_ab_arg
            kappas_ba_arg[i] = kappa_ba_arg
            kappas_ab[i] = kappa_ab
            kappas_ba[i] = kappa_ba

        del posterior_samples  # save some space :')

        return kappas_ab_arg, kappas_ab, kappas_ba_arg, kappas_ba, bins_ab, ratios_ab, bins_ba, ratios_ba

    @staticmethod
    def calc_individ_fracs(kappa_ab, kappa_ba):
        den = 1 - kappa_ab * kappa_ba
        fa = (1 - kappa_ab) / den
        fb = (kappa_ba - kappa_ab * kappa_ba) / den
        return fa, fb

    # note: this is unused in plotting, but here in case you want to take a look at the fractions
    def calc_fracs_from_kappa(self, kappas_ab, kappas_ba):
        kappa_ab, kappa_ab_std = np.mean(kappas_ab), np.std(kappas_ab)
        kappa_ba, kappa_ba_std = np.mean(kappas_ba), np.std(kappas_ba)

        fa, fb = self.calc_individ_fracs(kappa_ab, kappa_ba)

        den_err = np.sqrt(np.square(kappa_ba_std / kappa_ba) + np.square(kappa_ba_std / kappa_ab))
        fa_std = np.sqrt(np.square(kappa_ab_std / kappa_ab) + np.square(den_err))
        fb_num_err = np.sqrt(np.square(kappa_ba_std) + np.square(den_err))
        fb_std = np.sqrt(np.square(fb_num_err) + np.square(den_err))

        return fa, fb, fa_std, fb_std

    def calc_topics(self, kappas_ab, kappas_ba):
        def get_topic_and_err(pa, pa_errs, pb, pb_errs, kappa, kappa_errs):
            topic = (pa - kappa*pb)/(1-kappa)
            topic_errs = np.sqrt((pa - pb)**2 * kappa_errs**2 + (1 - kappa)**2 *
                                 (pa_errs**2 + kappa**2 * pb_errs**2)) / (1 - kappa)**2
            return topic, topic_errs

        kappa_ab, kappa_ab_std = np.mean(kappas_ab), np.std(kappas_ab)
        kappa_ba, kappa_ba_std = np.mean(kappas_ba), np.std(kappas_ba)

        topic1, topic1_err = get_topic_and_err(pa=self.data_obj.sample1.hist, pa_errs=self.data_obj.sample1.hist_error,
                                               pb=self.data_obj.sample2.hist, pb_errs=self.data_obj.sample2.hist_error, kappa=kappa_ab, kappa_errs=kappa_ab_std)
        topic2, topic2_err = get_topic_and_err(pa=self.data_obj.sample2.hist, pa_errs=self.data_obj.sample2.hist_error,
                                               pb=self.data_obj.sample1.hist, pb_errs=self.data_obj.sample1.hist_error, kappa=kappa_ba, kappa_errs=kappa_ba_std)

        return topic1, topic1_err, topic2, topic2_err

    def calc_substructure(self, substructure, kappa_ab_mean, kappa_ba_mean, kappa_ab_std, kappa_ba_std):
        def get_error(rho, kappa_mean, kappa_std, val1, err1, val2, err2):
            try:
                if not val2:
                    return np.nan
                kp_err = np.sqrt(np.square(kappa_std / kappa_mean) + np.square(err2 / val2))
                num_err = np.sqrt(np.square(err1) + np.square(kp_err * abs(kappa_mean * val2)))
                err = np.sqrt(np.square(num_err/(val1 - kappa_mean * val2)) + np.square(kappa_std / (1 - kappa_mean)))
                return abs(err * rho)
            except:
                return np.nan

        print(substructure)

        n_bins = config.SUBSTRUCTURES[substructure]["n_bins"]

        quark_vals = np.zeros((n_bins, 2))  # y, y_err
        gluon_vals = np.zeros((n_bins, 2))
        dijet_vals = np.zeros((n_bins, 2))
        photonjet_vals = np.zeros((n_bins, 2))
        x = np.zeros((n_bins, 2))
        n = [0, 0, 0, 0]  # quark n, gluon n, photonjet n, dijet n

        file_path = f'./substructure/input/{substructure[4:]}_jetpt{self.data_obj.min_pt}{self.data_obj.max_pt}_trackpt0_{self.data_obj.sample_type}.csv'
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:

                if config.SUBSTRUCTURES[substructure]["req_string_label"] not in row[0]:
                    continue

                if 'photonjet' in row[0].lower():  # photonjet
                    photonjet_vals[i] = np.array([float(row[3]), float(row[4])])
                    i = (i + 1) % n_bins
                    n[2] += float(row[3])
                elif 'quark' in row[0]:
                    # quark
                    quark_vals[i] = np.array([float(row[3]), float(row[4])])
                    i = (i + 1) % n_bins
                    n[0] += float(row[3])
                elif 'gluon' in row[0]:
                    # gluon
                    gluon_vals[i] = np.array([float(row[3]), float(row[4])])
                    i = (i + 1) % n_bins
                    n[1] += float(row[3])
                else:  # dijet
                    # x
                    x[i] = np.array([float(row[1]), float(row[2])])
                    dijet_vals[i] = np.array([float(row[3]), float(row[4])])
                    i = (i + 1) % n_bins
                    n[3] += float(row[3])

        # perform normalizations
        if substructure in ["jet-mass", "jet-splitting"]:
            bin_width = x[0, 0] * 2

            print(bin_width, n[2])
            quark_vals = quark_vals/n[0]/bin_width
            gluon_vals = gluon_vals/n[1]/bin_width
            photonjet_vals = photonjet_vals/n[2]/bin_width
            dijet_vals = dijet_vals/n[3]/bin_width

        # calculating topic 1 and topic 2
        topic1_vals = np.zeros((n_bins, 2))
        topic2_vals = np.zeros((n_bins, 2))

        topic1_vals[:, 0] = (photonjet_vals[:, 0] - kappa_ab_mean * dijet_vals[:, 0]) / (1 - kappa_ab_mean)
        topic2_vals[:, 0] = (dijet_vals[:, 0] - kappa_ba_mean * photonjet_vals[:, 0]) / (1 - kappa_ba_mean)

        for i in range(n_bins):
            # error calculations
            topic1_vals[i][1] = get_error(topic1_vals[i][0], kappa_ab_mean, kappa_ab_std,
                                          photonjet_vals[i][0], photonjet_vals[i][1], dijet_vals[i][0], dijet_vals[i][1])
            topic2_vals[i][1] = get_error(topic2_vals[i][0], kappa_ba_mean, kappa_ba_std,
                                          dijet_vals[i][0], dijet_vals[i][1], photonjet_vals[i][0], photonjet_vals[i][1])

        return x, quark_vals, gluon_vals, photonjet_vals, dijet_vals, topic1_vals, topic2_vals

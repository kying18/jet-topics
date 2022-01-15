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

    def get_kappas(self, samples):
        # randomly sample "nkappa" points from the posterior on which to extract kappa
        all_index_tuples = [(i, j) for i in range(self.burn_in, len(samples)) for j in range(len(samples[0]))]
        index_tuples = random.sample(all_index_tuples, self.nkappa)
        posterior_samples = np.array([[samples[tup[0], tup[1], i] for i in range(config.NDIM)] for tup in index_tuples])

        # we want to only extract kappa where hist1 or hist2 has data
        or_mask = (self.data_obj.sample1.hist_unnorm > 0) | (self.data_obj.sample2.hist_unnorm > 0)

        kappa_AB, kappa_BA = np.zeros(len(posterior_samples)), np.zeros(len(posterior_samples))

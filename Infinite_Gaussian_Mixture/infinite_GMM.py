'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random

import numpy as np
import scipy as sc
import scipy.special

import Sampling as sampling_package
import Sampling.adaptive_rejection_sampling


class infinite_GMM:
    def __init__(self, X, initialZ=None, initial_number_class=np.random.randint(2, 14, size=1)[0]):
        self.X = X
        self.mean_sample = X.mean()
        self.variance_sample = np.mean(abs(X - X.mean()) ** 2)[0]
        self.lambda_prior = sc.random.normal(loc=self.mean_sample, scale=self.variance_sample, size=1)[0]
        self.r_prior = sc.random.gamma(shape=1, scale=1. / self.variance_sample, size=1)[0]
        self.beta = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.w = sc.random.gamma(shape=1, scale=self.variance_sample, size=1)[0]
        self.alpha = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.means = []
        self.S = []
        if initialZ is None:
            self.Z = np.random.randint(0, initial_number_class, size=len(X))
        else:
            self.Z = np.array(initialZ.copy())

        for k in range(len(np.unique(self.Z))):
            Xk = self.X[np.where(self.Z == k)]
            meank = Xk.mean()
            sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)[0]
            self.means.append(meank)
            self.S.append(1. / sigmak)
        self.means = np.array(self.means)
        self.S = np.array(self.S)

    @staticmethod
    def __h_log_beta(log_beta, *args):
        S = args[0]
        w = args[1]
        K = len(S)
        return -K * sc.special.gammaln(log_beta / 2.) - 1. / (2 * log_beta) + (K * log_beta - 3) / 2. * (
            np.log(log_beta / 2.)) + log_beta / 2. * (np.sum(np.log(S * w) - S * w))

    @staticmethod
    def __h_derivative_beta(log_beta, *args):
        S = args[0]
        w = args[1]
        K = len(S)
        return -K * sc.special.digamma(log_beta / 2) / 2. + 1. / (2 * (log_beta ** 2)) + K / 2. * (
            np.log(log_beta / 2.)) + \
               (K * log_beta - 3) / (2. * log_beta) + 1 / 2. * (np.sum(np.log(S * w) - S * w))

    @staticmethod
    def __h_log_alpha(log_alpha, *args):
        K = args[0]
        N = args[1]
        return np.log(K - 3. / 2) + np.log(np.abs(log_alpha)) - 1. / (2 * log_alpha) + sc.special.gammaln(
            log_alpha) - sc.special.gammaln(N + log_alpha)

    @staticmethod
    def __h_derivative_alpha(log_alpha, *args):
        # K = args[0] # K is not used in the derivative
        N = args[1]
        return (2 * log_alpha - 1) / (2 * log_alpha ** 2) + sc.special.digamma(log_alpha) - sc.special.digamma(
            N + log_alpha)

    def __sample_means_precision(self):
        K = len(self.means)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.X[np.where(self.Z == k)]
            meank = Xk.mean()
            sk = 1. / np.mean(abs(Xk - Xk.mean()) ** 2)[0]
            Nk = len(Xk)
            mean_value = (meank * Nk * sk + self.lambda_prior * self.r_prior) / (Nk * sk + self.r_prior)
            sigma_value = 1. / (Nk * sk + self.r_prior)
            meank = np.random.normal(loc=mean_value, scale=sigma_value, size=1)[0]
            new_means.append(meank)

            # sampling sk
            shape_parameter = self.beta + Nk
            scale_paremeter = 1. / ((1. / (self.beta + Nk)) * (self.w * self.beta + np.sum((Xk - meank) ** 2)))
            new_S.append(sc.random.gamma(shape=shape_parameter, scale=scale_paremeter, size=1)[0])

        self.means = np.array(new_means)
        self.S = np.array(new_S)

    def __sample_lambda_r_priors(self):
        K = len(self.means)
        mean_value_lambda = (self.mean_sample * (1. / self.variance_sample) + self.r_prior * np.sum(self.means)) / (
                    1. / self.variance_sample + K * self.r_prior)
        variance_value_lambda = 1. / ((1. / self.variance_sample) + K * self.r_prior)
        self.lambda_prior = sc.random.normal(loc=mean_value_lambda, scale=variance_value_lambda, size=1)[0]

        shape_value_r = K + 1
        scale_value_r = 1. / (1. / (K + 1) * (self.variance_sample + np.sum(self.means - self.lambda_prior) ** 2))
        self.r_prior = sc.random.gamma(shape=shape_value_r, scale=scale_value_r, size=1)[0]

    def __sample_w_beta_priors(self):
        K = len(self.means)
        shape_value = K * self.beta + 1
        scale_value = 1. / (1. / (K * self.beta + 1) * (1. / self.variance_sample + self.beta * np.sum(self.S)))

        self.w = sc.random.gamma(shape=shape_value, scale=scale_value, size=1)[0]

        Tk = [0.001, 100]
        x0 = 0.0008
        xk_plus_1 = 500

        ars = sampling_package.adaptive_rejection_sampling.ARS(Tk, x0, xk_plus_1, 10000, self.__h_log_beta,
                                                               self.__h_derivative_beta, self.S, self.w)

        numbres_beta_sampling = 100
        beta_samplings = ars.perform_ARS(numbres_beta_sampling, False)

        self.beta = beta_samplings[np.random.randint(0, numbres_beta_sampling, size=1)[0]]

    def __sample_Z(self):
        N = len(self.X)
        order_sampling_Z = random.shuffle(range(len(self.Z)))
        # shuffle the index to more fastest converge
        for i in order_sampling_Z:
            K = len(self.means)
            new_means_precision = {}
            probability_of_belongs_class = []
            for k in range(K):
                Nk = len(np.where(self.Z == k)[0])
                if self.Z[i] == k:
                    Nk -= 1

                if Nk > 0:
                    prob = float(Nk) / (N - 1 + self.alpha) * np.sqrt(self.S[k]) * np.exp(
                        -self.S[k] * (self.X[i] - self.means[k]) ** 2 / 2.)
                else:
                    sk = sc.random.normal(loc=self.lambda_prior, scale=1. / self.r_prior, size=1)[0]
                    meank = sc.random.gamma(shape=self.beta, scale=self.w, size=1)[0]

                    new_means_precision[k] = [meank, sk]
                    prob = float(1) / (N - 1 + self.alpha) * np.sqrt(sk) * np.exp(
                        -sk * (self.X[i] - meank) ** 2 / 2.)

                probability_of_belongs_class.append(prob)
            sk = sc.random.normal(loc=self.lambda_prior, scale=1. / self.r_prior, size=1)[0]
            meank = sc.random.gamma(shape=self.beta, scale=self.w, size=1)[0]

            new_means_precision[K] = [meank, sk]
            prob = float(1) / (N - 1 + self.alpha) * np.sqrt(sk) * np.exp(
                -sk * (self.X[i] - meank) ** 2 / 2.)
            probability_of_belongs_class.append(prob)

            probability_of_belongs_class = np.array(probability_of_belongs_class) / np.sum(probability_of_belongs_class)

            u = np.random.uniform(size=1)[0]
            actualProb = 0
            for k, p in enumerate(probability_of_belongs_class):
                actualProb += p
                if actualProb >= u:
                    if k in new_means_precision:  # If is a new mean an precision.
                        if k < K:
                            self.means[k] = new_means_precision[k][0]
                            self.S[k] = new_means_precision[k][1]
                        else:
                            self.means = np.append(self.means, new_means_precision[k][0])
                            self.S = np.append(self.S, new_means_precision[k][1])
                    self.Z[i] = k
                    break

        if len(self.means) != len(np.unique(self.Z)):
            print "Error"

    def __sample_alpha(self):
        N = len(self.X)
        K = len(self.means)

        Tk = [0.001, 100]
        x0 = 0.0008
        xk_plus_1 = 500

        ars = sampling_package.adaptive_rejection_sampling.ARS(Tk, x0, xk_plus_1, 10000, self.__h_log_alpha,
                                                               self.__h_derivative_alpha, K, N)

        numbres_alpha_sampling = 100
        alpha_sampling = ars.perform_ARS(numbres_alpha_sampling, False)

        self.alpha = alpha_sampling[np.random.randint(0, numbres_alpha_sampling, size=1)[0]]

    def learn_GMM(self, number_of_loops):
        for i in range(number_of_loops):
            self.__sample_means_precision()
            self.__sample_lambda_r_priors()
            self.__sample_w_beta_priors()
            self.__sample_Z()
            self.__sample_alpha()

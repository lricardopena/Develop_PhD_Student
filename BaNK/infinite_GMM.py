'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random

import numpy as np
import scipy as sc
import scipy.special
import scipy.stats
from sklearn.cluster import KMeans


class infinite_GMM:
    def __init__(self, X, initialZ=None, initial_number_class=np.random.randint(2, 14, size=1)[0]):
        self.X = X
        self.oneDimension = (len(X.shape) == 1)

        self.alpha = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]

        if self.oneDimension:
            self.D = 1
            self.beta = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
            self.mean_sample = X.mean()
            self.variance_sample = float(np.mean(abs(X - self.mean_sample) ** 2))
            self.lambda_prior = sc.random.normal(loc=self.mean_sample, scale=np.sqrt(self.variance_sample), size=1)[0]
            self.r_prior = sc.random.gamma(shape=1, scale=1. / self.variance_sample, size=1)[0]
            self.w = sc.random.gamma(shape=1, scale=self.variance_sample, size=1)[0]

            if initialZ is None:
                self.Z = np.random.randint(0, initial_number_class, size=len(X))
            else:
                self.Z = np.array(initialZ.copy())
        else:
            self.D = X.shape[1]
            self.beta = 1. / sc.random.gamma(shape=1./self.D, scale=1, size=1)[0] + self.D - 1
            self.mean_sample = X.mean(axis=0)
            self.variance_sample = np.cov(X.T)
            self.inverse_variance_sample = np.linalg.inv(self.variance_sample)
            N = len(X)
            self.lambda_prior = sc.random.multivariate_normal(mean=self.mean_sample, cov=self.variance_sample, size=1)
            self.r_prior = sc.stats.wishart.rvs(N-1, self.inverse_variance_sample, size=1)
            self.w = sc.stats.wishart.rvs(N-1, self.variance_sample, size=1)
            if initialZ is None:
                kmeans = KMeans(n_clusters=initial_number_class).fit(X)
                self.Z = np.array(kmeans.labels_)
            else:
                self.Z = np.array(initialZ.copy())

        self.means = []
        self.S = []
        for k in range(len(np.unique(self.Z))):
            Xk = self.X[np.where(self.Z == k)]

            if self.oneDimension:
                meank = Xk.mean()
                sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)
                self.S.append(1. / sigmak)
            else:
                meank = Xk.mean(axis=0)
                self.S.append(np.linalg.inv(np.cov(Xk.T)))
            self.means.append(meank)

        self.means = np.array(self.means)
        self.S = np.array(self.S)

    @staticmethod
    def __sample_scaled_inverse_chi_square(scale, shape):
        return sc.stats.invgamma.rvs(a=scale/2., scale=scale*shape/2., size=1)[0]

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
        return (K - 3./2) * np.log(log_alpha) - 1/(2*log_alpha) + sc.special.gammaln(log_alpha) - sc.special.gammaln(N + log_alpha)

    @staticmethod
    def __h_derivative_alpha(log_alpha, *args):
        K = args[0]
        N = args[1]
        sum = 0
        for i in range(N):
            sum += 1./(N+log_alpha-i)

        return (K - 3./2)/log_alpha + 1./(2*log_alpha**2) + sum

    def get_covariance_matrix(self):
        covarianceMatrices = []
        for cov in self.S:
            cov = np.linalg.inv(cov)
            covarianceMatrices.append(cov)
        return np.array(covarianceMatrices)

    def get_weights(self):
        K = len(self.means)
        N = float(len(self.X))
        pik = np.zeros(K)
        for k in range(K):
            pik[k] = len(np.where(self.Z == k)[0])/N

        return pik

    def get_variance_1d(self):
        return np.sqrt(1./self.S)

    def __sample_means_precision_1_dimension(self):
        K = len(self.means)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.X[np.where(self.Z == k)]
            meank = Xk.mean()
            sk = self.S[k]
            Nk = len(Xk)
            mean_value = (meank * Nk * sk + self.lambda_prior * self.r_prior) / (Nk * sk + self.r_prior)
            sigma_value = 1. / (Nk * sk + self.r_prior)
            try:
                meank = np.random.normal(loc=mean_value, scale=np.sqrt(sigma_value), size=1)[0]
            except ValueError:
                print("some")
            new_means.append(meank)

            # sampling sk
            shape_parameter = self.beta + Nk
            # scale_paremeter = (self.w * self.beta + np.sum((Xk - meank) ** 2))/(self.beta + Nk)
            # scale_parameter = np.linalg.inv((S_omega + np.linalg.inv(self.w * self.beta)))
            # scale_paremeter = 1./((1./(self.w * self.beta) + np.sum((Xk - meank) ** 2))/(self.beta + Nk))
            scale_paremeter = (1. / (self.beta + Nk)) * (self.w * self.beta + np.sum((Xk - meank) ** 2))
            sk = 1./self.__sample_scaled_inverse_chi_square(shape_parameter, scale_paremeter)
            new_S.append(sk)

        self.means = np.array(new_means)
        self.S = np.array(new_S)


    def __sample_means_precision(self):
        if self.oneDimension:
            self.__sample_means_precision_1_dimension()
        else:
            K = len(self.means)
            new_means = []
            new_S = []

            for k in range(K):
                # sampling muk
                Xk = self.X[np.where(self.Z == k)]
                meank = Xk.mean(axis=0)
                sk = self.S[k]
                Nk = len(Xk)
                try:
                    sigma_value = np.linalg.inv(Nk * sk + self.r_prior)
                except np.linalg.LinAlgError:
                    #If is singular we add some value
                    sigma_value = np.linalg.inv(Nk * sk + self.r_prior + 0.001 * np.identity(self.D))
                mean_value = (Nk*meank.dot(sk) + self.lambda_prior.dot(self.r_prior)).dot(sigma_value).reshape(self.D,)

                newmeank = sc.stats.multivariate_normal.rvs(mean=mean_value, cov=sigma_value, size=1)

                new_means.append(newmeank)
                # sampling sk
                shape_parameter = Nk-1+self.beta+self.D
                S = np.zeros_like(self.w)

                for x in Xk:
                    S += (x - meank).reshape(self.D, 1).dot((x - meank).reshape(1, self.D))

                scale_parameter = np.linalg.inv((S+np.linalg.inv(self.w*self.beta)))
                newSk = sc.stats.wishart.rvs(shape_parameter,scale_parameter,1)
                new_S.append(newSk)

            self.means = np.array(new_means)
            self.S = np.array(new_S)

    def __sample_lambda_r_priors(self):
        K = len(self.means)
        if self.oneDimension:
            mean_value_lambda = (self.mean_sample * (1. / self.variance_sample) + self.r_prior * np.sum(self.means)) / (
                        1. / self.variance_sample + K * self.r_prior)
            variance_value_lambda = 1. / ((1. / self.variance_sample) + K * self.r_prior)
            self.lambda_prior = sc.random.normal(loc=mean_value_lambda, scale=np.sqrt(variance_value_lambda), size=1)[0]

            shape_value_r = K + 1
            scale_value_r = 1. / (1. / (K + 1) * (self.variance_sample + np.sum(self.means - self.lambda_prior) ** 2))
            self.r_prior = sc.random.gamma(shape=shape_value_r, scale=scale_value_r, size=1)[0]
        else:
            variance_value_lambda = np.linalg.inv(self.inverse_variance_sample + K * self.r_prior)
            mean_value_lambda = (self.mean_sample.dot(self.inverse_variance_sample) + self.r_prior.dot(
                np.sum(self.means, axis=0))).dot(variance_value_lambda)

            self.lambda_prior = sc.stats.multivariate_normal.rvs(mean=mean_value_lambda, cov=variance_value_lambda, size=1)

            shape_value_r = K + 1
            sum_result = np.zeros_like(variance_value_lambda)
            for row in self.means - self.lambda_prior:
                sum_result += row.reshape(self.D, 1).dot(row.reshape(1, self.D))
            scale_value_r =  1. / (K + 1)*np.linalg.inv(self.variance_sample + sum_result)

            self.r_prior = sc.stats.wishart.rvs(shape_value_r, scale_value_r, size=1)

    def __sample_w_beta_priors(self):

        K = len(self.means)
        if self.oneDimension:
            shape_value = K * self.beta + 1
            scale_value = (1. / (K * self.beta + 1) * (1. / self.variance_sample + self.beta * np.sum(self.S)))

            if scale_value <= 0:
                print("error")

            self.w = sc.random.gamma(shape=shape_value, scale=1./scale_value, size=1)[0]

            # Tk = [0.001, 100]
            # x0 = 0.0000000000008
            # xk_plus_1 = 500
            #
            # ars = sampling_package.adaptive_rejection_sampling.ARS(Tk, x0, xk_plus_1, 10000, self.__h_log_beta,
            #                                                        self.__h_derivative_beta, self.S_omega, self.w)
            #
            # numbres_beta_sampling = 10
            # beta_samplings = ars.perform_ARS(numbres_beta_sampling, False)
            # beta_samplings = np.exp(beta_samplings)
            # self.beta = beta_samplings[np.random.randint(0, numbres_beta_sampling, size=1)[0]]


        else:

            shape_value = K * self.beta + self.D
            scale_value = (K * self.beta + self.D) * np.linalg.inv((self.inverse_variance_sample + self.beta * np.sum(self.S, axis=0)))
            self.w = sc.stats.invwishart.rvs(shape_value, scale_value, size=1)

    def __sample_Z_1_dimension(self):
        N = len(self.X)
        order_sampling_Z = range(len(self.Z))
        random.shuffle(order_sampling_Z)
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
                    prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + 1./2*np.log(self.S[k]) - (self.S[k]*(self.X[i] - self.means[k])**2)/2.
                else:
                    meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
                    sk = sc.random.gamma(shape=self.beta, scale=1./self.w, size=1)[0]

                    new_means_precision[k] = [meank, sk]
                    prob = - np.log(N - 1 + self.alpha) + 1./2*np.log(sk) - (sk*(self.X[i] - meank)**2)/2.

                probability_of_belongs_class.append(prob)
            meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
            sk = sc.random.gamma(shape=self.beta, scale=1./self.w, size=1)[0]

            new_means_precision[K] = [meank, sk]
            prob = - np.log(N - 1 + self.alpha) + 1./2*np.log(sk) - (sk*(self.X[i] - meank)**2)/2.
            probability_of_belongs_class.append(prob)
            if np.sum(np.exp(probability_of_belongs_class)) > 0:
                probability_of_belongs_class = np.exp(probability_of_belongs_class)
            else:
                probability_of_belongs_class -= np.min(probability_of_belongs_class)
            probability_of_belongs_class = np.cumsum(
                np.array(probability_of_belongs_class) / np.sum(probability_of_belongs_class))

            u = np.random.uniform(size=1)[0]
            for k, actualProb in enumerate(probability_of_belongs_class):
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

    def __sample_Z(self):
        if self.oneDimension:
            self.__sample_Z_1_dimension()
            if len(self.means) != len(np.unique(self.Z)):
                k = 0
                while k < len(self.means):
                    if len(np.where(self.Z == k)[0]) <= 0:
                        self.means = np.delete(self.means, k)
                        self.S = np.delete(self.S, k)
                        self.Z[np.where(self.Z > k)[0]] -= 1
                    else:
                        k += 1
        else:
            N = len(self.X)
            order_sampling_Z = range(len(self.Z))
            random.shuffle(order_sampling_Z)
            r_prior_inverse = np.linalg.inv(self.r_prior)
            w_inverse = np.linalg.inv(self.w)
            # shuffle the index to more fastest converge

            inverseS = []
            for k in range(len(self.means)):
                try:
                    inverseS.append(np.linalg.inv(self.S[k]))
                except np.linalg.LinAlgError:
                    inverseS.append(np.linalg.inv(self.S[k] + 0.0001 * np.identity(self.D)))


            for i in order_sampling_Z:
                K = len(self.means)
                new_means_precision = {}
                probability_of_belongs_class = []
                for k in range(K):
                    Nk = len(np.where(self.Z == k)[0])
                    if self.Z[i] == k:
                        Nk -= 1

                    if Nk > 0:
                        try:
                            prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.X[i], mean=self.means[k].reshape(self.D,), cov=inverseS[k])
                        except (np.linalg.LinAlgError, ValueError):
                            self.S[k] += 0.01 * np.identity(self.D)
                            inverseS[k] = np.linalg.inv(self.S[k])
                            prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                                self.X[i], mean=self.means[k].reshape(self.D, ), cov=inverseS[k])

                    else:
                        meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)
                        sk = sc.stats.wishart.rvs(self.beta + self.D, w_inverse, size=1)

                        try:
                            prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.X[i], meank, cov=np.linalg.inv(sk))
                        except np.linalg.LinAlgError:
                            sk += 0.01* np.identity(self.D)
                            prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.X[i], meank,
                                                                                                      cov=np.linalg.inv(
                                                                                                          sk))

                        new_means_precision[k] = [meank, sk]

                    probability_of_belongs_class.append(prob)
                meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)

                sk = sc.stats.wishart.rvs(self.beta + self.D, w_inverse, size=1)

                try:
                    prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.X[i], meank, cov=np.linalg.inv(sk))
                except (ValueError, np.linalg.LinAlgError):
                    sk += 0.001 * np.identity(self.D)
                    prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.X[i], meank, cov=np.linalg.inv(sk))

                new_means_precision[K] = [meank, sk]

                probability_of_belongs_class.append(prob)
                if np.sum(np.exp(probability_of_belongs_class)) > 0:
                    probability_of_belongs_class = np.exp(probability_of_belongs_class)
                else:
                    probability_of_belongs_class -= np.min(probability_of_belongs_class)
                probability_of_belongs_class = np.cumsum(
                    np.array(probability_of_belongs_class) / np.sum(probability_of_belongs_class))

                u = np.random.uniform(size=1)[0]
                for k, actualProb in enumerate(probability_of_belongs_class):
                    if actualProb >= u:
                        if k in new_means_precision:  # If is a new mean an precision.
                            if k < K:
                                try:
                                    self.means[k] = new_means_precision[k][0].reshape(self.D, 1)
                                except ValueError:
                                    self.means[k] = new_means_precision[k][0].reshape(self.D, )
                                self.S[k] = new_means_precision[k][1]
                                inverseS[k] = np.linalg.inv(self.S[k])
                            else:
                                self.means = np.concatenate((self.means.reshape(self.means.shape[0], self.D, 1), new_means_precision[k][0].reshape(1, self.D, 1)), axis=0)
                                self.S = np.concatenate((self.S, new_means_precision[k][1].reshape(1, 2, 2)))
                                inverseS.append(np.linalg.inv(new_means_precision[k][1]))
                        self.Z[i] = k
                        break

            if len(self.means) != len(np.unique(self.Z)):
                k = 0
                while k < len(self.means):
                    if len(np.where(self.Z == k)[0]) <= 0:
                        self.means = np.delete(self.means, k, axis=0)
                        self.S = np.delete(self.S, k, axis=0)
                        del inverseS[k]
                        self.Z[np.where(self.Z > k)[0]] -= 1
                    else:
                        k += 1

    def learn_GMM(self, number_of_loops):
        for i in range(number_of_loops):
            self.__sample_means_precision()
            self.__sample_lambda_r_priors()
            self.__sample_w_beta_priors()
            self.__sample_Z()
            if len(self.S) < 2:
                print("error")
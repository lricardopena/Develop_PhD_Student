import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random

import numpy as np
import scipy as sc
import scipy.special
import scipy.stats
import Sampling as sampling_package
import Sampling.adaptive_rejection_sampling
from sklearn.cluster import KMeans
import infinite_GMM


class bank:
    def __init__(self, X, Y, M, realomegas = None):
        self.X = X
        self.Y = Y
        self.M = M
        self.N = len(Y)
        if not realomegas is None:
            self.real_log_error = self.__compute_log_model_evidence(realomegas)
        self.oneDimension = (len(X.shape) == 1)
        K = np.random.randint(2, 14, size=1)[0]
        if self.oneDimension:
            self.D = 1
            self.omegas = np.random.rand(M)
            self.beta = np.random.rand(2 * self.M)
        else:
            self.D = X.shape[1]
            self.omegas = np.random.rand(M, self.D)
            self.beta = np.random.rand(2 * self.M, self.D)

        self.alpha = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.beta_prior = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.mean_sample = self.omegas.mean(axis=0)
        self.variance_sample = float(np.mean(abs(self.omegas - self.mean_sample) ** 2))
        self.inverse_variance_sample = 1./self.variance_sample
        self.lambda_prior = sc.random.normal(loc=self.mean_sample, scale=np.sqrt(self.variance_sample), size=1)[0]
        self.r_prior = sc.random.gamma(shape=1, scale=1. / self.variance_sample, size=1)[0]
        self.w_prior = sc.random.gamma(shape=1, scale=self.variance_sample, size=1)[0]
        self.Z = np.random.randint(0, K, size=self.M)

        self.means = []
        self.S = []
        for k in range(len(np.unique(self.Z))):
            Xk = self.omegas[np.where(self.Z == k)[0]]

            meank = Xk.mean()
            sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)
            self.S.append(1. / sigmak)
            self.means.append(meank)

        self.means = np.array(self.means)
        self.S = np.array(self.S)

        beta_act, sigma_ep = self.__returnBeta_sigma(self.__matrix_phi(self.omegas), 1)
        self.beta = beta_act
        self.sigma_e = sigma_ep

    @staticmethod
    def __sample_scaled_inverse_chi_square(scale, shape):
        return sc.stats.invgamma.rvs(a=scale / 2., scale=scale * shape / 2., size=1)[0]

    @staticmethod
    def __multivariate_t_rvs(m, S, df=np.inf, n=1):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        n : int
            number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        m = np.asarray(m)
        d = len(m)
        if df == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        return m + z / np.sqrt(x)[:, None]  # same output format as random.multivariate_normal

    def __sample_w_beta_priors(self):
        K = len(self.means)
        if self.oneDimension:
            shape_value = K * self.beta_prior + 1
            scale_value = (1. / (K * self.beta_prior + 1) * (1. / self.variance_sample + self.beta_prior * np.sum(self.S)))
            if scale_value <= 0:
                print("error")
            self.w_prior = sc.random.gamma(shape=shape_value, scale=1./scale_value, size=1)[0]

        else:
            shape_value = K * self.beta_prior + self.D
            scale_value = (K * self.beta_prior + self.D) * np.linalg.inv((self.inverse_variance_sample + self.beta_prior * np.sum(self.S, axis=0)))
            self.w_prior = sc.stats.invwishart.rvs(shape_value, scale_value, size=1)

    def __sampling_Z(self):
        N = self.M
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
                    prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(self.S[k]) - (
                                self.S[k] * (self.omegas[i] - self.means[k]) ** 2) / 2.
                else:
                    meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
                    sk = sc.random.gamma(shape=self.beta_prior, scale=1. / self.w_prior, size=1)[0]

                    new_means_precision[k] = [meank, sk]
                    prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (sk * (self.X[i] - meank) ** 2) / 2.

                probability_of_belongs_class.append(prob)
            meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
            sk = sc.random.gamma(shape=self.beta_prior, scale=1. / self.w_prior, size=1)[0]

            new_means_precision[K] = [meank, sk]
            prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (sk * (self.omegas[i] - meank) ** 2) / 2.
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
        if len(self.means) != len(np.unique(self.Z)):
            k = 0
            while k < len(self.means):
                if len(np.where(self.Z == k)[0]) <= 0:
                    self.means = np.delete(self.means, k)
                    self.S = np.delete(self.S, k)
                    self.Z[np.where(self.Z > k)[0]] -= 1
                else:
                    k += 1

    def __sampling_mu_sigma(self):
        K = len(self.means)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.omegas[np.where(self.Z == k)]
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
            shape_parameter = self.beta_prior + Nk
            scale_paremeter = (1. / (self.beta_prior + Nk)) * (self.w_prior * self.beta_prior + np.sum((Xk - meank) ** 2))
            sk = 1. / self.__sample_scaled_inverse_chi_square(shape_parameter, scale_paremeter)
            new_S.append(sk)

        self.means = np.array(new_means)
        self.S = np.array(new_S)

    def __returnBeta_sigma(self, X, k):
        N = len(X)
        Q, R = np.linalg.qr(X)
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.inv(R + 0.0001 * np.identity(R.shape[0]))

        # if exist any nan or inf or -inf
        if np.any(np.isinf(Rinv)) or np.any(np.isnan(Rinv)):
            Rinv = np.nan_to_num(Rinv)
        # Vbeta = np.linalg.inv(X.T.dot(X))
        Vbeta = Rinv.dot(Rinv.T)
        # betahat = Vbeta.dot(X.T.dot(Y))
        betahat = Rinv.dot(Q.T.dot(self.Y))
        sSquare = 1. / (N - k) * (self.Y - X.dot(betahat)).T.dot(self.Y - X.dot(betahat))
        sigma = self.__sample_scaled_inverse_chi_square(N - k, sSquare)
        variance = Vbeta * (sigma ** 2)
        if np.any(np.isinf(variance)) or np.any(np.isnan(variance)):
            variance[variance == -np.inf] = -999999999
            variance[variance == np.inf] = 999999999
            variance[np.isnan(variance)] = 0.01

        try:
            betaProposal = np.random.multivariate_normal(betahat, variance)
            return betaProposal, sigma
        except:
            print("stop")
            # Checar cual de las dos
            # return betahat, sigma
            return Vbeta, sigma

    def get_Phi_X(self, X, omegas):
        return self.__matrix_phi_with_X(omegas, X)

    def predict_new_X(self, X, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        if miu0 is None:
            Phi_x = self.__matrix_phi(omegas)
            an = a0 + self.N/2.

            Lamda0 = 1./alpha_prior*np.identity(self.M*2 + 1)

            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, omegas)
            mean = Phi_x_new.dot(miun)
            Sigma = an/bn*(np.identity(len(X))+ Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
            df = 2*an
        else:
            miu0 = np.zeros(self.M * 2 + 1)
            Phi_x = self.__matrix_phi(omegas)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2 + 1)

            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, omegas)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
            df = 2 * an
        return self.__multivariate_t_rvs(mean, Sigma, df, n=len(X))[0]


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

            self.lambda_prior = sc.stats.multivariate_normal.rvs(mean=mean_value_lambda, cov=variance_value_lambda,
                                                                 size=1)

            shape_value_r = K + 1
            sum_result = np.zeros_like(variance_value_lambda)
            for row in self.means - self.lambda_prior:
                sum_result += row.reshape(self.D, 1).dot(row.reshape(1, self.D))
            scale_value_r = 1. / (K + 1) * np.linalg.inv(self.variance_sample + sum_result)

            self.r_prior = sc.stats.wishart.rvs(shape_value_r, scale_value_r, size=1)

    def __compute_log_model_evidence(self, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        '''
        :param omegas: the matrix with all w in W
        :param a0: loc prior for sigma value
        :param b0: scale prior for sigma value
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        '''
        if miu0 is None:
            Phi_x = self.__matrix_phi(omegas)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2 + 1)

            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))

            result = sc.special.gammaln(an) - sc.special.gammaln(a0) - a0 * np.log(b0) - (an * np.log(bn)) + 1./2 * ((self.M*2 + 1)*np.log(alpha_prior) - np.linalg.slogdet(Lamdan)[1])
        else:
            Phi_x = self.__matrix_phi(omegas)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2 + 1)

            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

            result = sc.special.gammaln(an) - sc.special.gammaln(a0) - a0 * np.log(b0) - (an * np.log(bn)) + 1. / 2 * (
                        (self.M * 2 + 1) * np.log(alpha_prior) - np.linalg.slogdet(Lamdan)[1])
        return result

    def __sample_omega(self):
        actual__log_error = self.__compute_log_model_evidence(self.omegas)
        for j in range(self.M):
            Zj = self.Z[j]
            w_proposal = sc.random.normal(self.means[Zj], np.sqrt(1./self.S[Zj]), size=1)[0]
            w_new = self.omegas.copy()
            w_new[j] = w_proposal
            new_log_error = self.__compute_log_model_evidence(w_new)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u: # Acept with certain probability
                    actual__log_error = new_log_error
                    self.omegas = w_new

    @staticmethod
    def phi_xi(x, w):
        argument = w.dot(x).T

        return np.concatenate(([1], np.cos(argument), np.sin(argument)))

    def __matrix_phi_with_X(self, omegas, X):
        means = []
        for x in X:
            means.append(self.phi_xi(x, omegas))
        return np.array(means)

    def __matrix_phi(self, omegas):
        means = []
        for x in self.X:
            means.append(self.phi_xi(x, omegas))
        return np.array(means)

    def f(self, omegas):
        Phi_x = self.__matrix_phi(omegas)

        means = np.array(Phi_x).dot(self.beta)
        Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
        return Y


    def get_beta_sigma_with_given_omega(self, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        #sample
        if miu0 is None:
            miu0 = np.zeros(self.M * 2 + 1)
        Phi_x = self.__matrix_phi(omegas)
        an = a0 + self.N/2.

        Lamda0 = alpha_prior*np.identity(self.M*2 + 1)

        Lamdan =  Phi_x.T.dot(Phi_x) + Lamda0

        miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))

        bn = b0 + 1./2*(self.Y.dot(self.Y) + miu0.T.dot(Lamda0).dot(miu0) - miun.T.dot(Lamdan).dot(miun))
        sigma_e = sc.stats.invgamma(an, bn)

        beta_sampled = self.__multivariate_t_rvs(miun, bn/an*Lamdan, 2*an)

        return beta_sampled, sigma_e

    def __sample_beta_sigma(self):
        self.beta, self.sigma_e = self.__returnBeta_sigma(self.__matrix_phi(self.omegas), 1)

    def get_omegas(self):
        return self.omegas

    def get_beta_sigma(self):
        return self.beta, self.sigma_e

    def __sample_priors(self):
        self.__sample_w_beta_priors()
        self.__sample_lambda_r_priors()
        self.mean_sample = self.omegas.mean(axis=0)
        self.variance_sample = float(np.mean(abs(self.omegas - self.mean_sample) ** 2))
        self.inverse_variance_sample = 1. / self.variance_sample

    def learn_kernel(self, number_swaps):
        for i in range(number_swaps):
            self.__sampling_Z()
            self.__sampling_mu_sigma()
            self.__sample_omega()
            self.__sample_priors()

        self.__sample_beta_sigma()

    def get_pik(self):
        K = len(self.means)
        N = float(len(self.omegas))
        pik = np.zeros(K)
        for k in range(K):
            pik[k] = len(np.where(self.Z == k)[0])/N

        return pik

    def get_covariance_matrix(self):
        covarianceMatrices = []
        for cov in self.S:
            cov = np.linalg.inv(cov)
            covarianceMatrices.append(cov)
        return np.array(covarianceMatrices)
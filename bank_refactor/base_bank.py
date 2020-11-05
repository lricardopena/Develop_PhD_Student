"""
This code is an implementation of Bayesian Nonparametrics Kernel Learning from Oliva 2015 (BaNK)
"""

import gc
import random
from typing import Union

import numpy as np
import scipy as sc
import scipy.special
from scipy.stats import beta
from sklearn.cluster import KMeans


class BaseBank:
    M: int
    Z: np.ndarray
    D: int
    alpha: float
    omegas: np.ndarray
    r_prior: Union[float, np.ndarray]
    w_prior: Union[float, np.ndarray]
    means: np.ndarray
    S: np.ndarray
    beta: np.ndarray
    mean_sample: Union[float, np.ndarray]
    variance_sample: Union[float, np.ndarray]
    inverse_variance_sample: Union[float, np.ndarray]
    lambda_prior: Union[float, np.ndarray]

    def __init__(self, X, Y, M, bias=0):
        self.X = X
        self.N = len(Y)

        bias = int(round(bias))

        if bias > 1:
            bias = 1
        elif bias < 0:
            bias = 0

        self.bias = bias
        if len(Y.shape) == 1:
            self.Y = Y
        else:
            self.Y = Y.reshape(len(Y), )

        self.M = M
        self.constant = 1. / np.sqrt(M)

        self.oneDimension = (len(X.shape) == 1)
        if not self.oneDimension:
            self.D = X.shape[1]
            if self.D == 1:
                self.oneDimension = True

        K = np.random.randint(2, 14, size=1)[0]

        if self.oneDimension:
            self.initialize_one_dimension(K)
        else:
            self.initialize_multidimension(K)

        self.alpha = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.beta_prior = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.sigma_e = 999999.
        del X, Y

    def initialize_one_dimension(self, K):
        self.D = 1
        self.X = self.X.reshape(self.N, 1)
        self.omegas = np.random.rand(self.M)
        self.beta = np.random.rand(2 * self.M + self.bias)
        # self.mean_sample = self.omegas.mean(axis=0)
        self.mean_sample = 0

        self.variance_sample = 99999.
        self.inverse_variance_sample = 1. / self.variance_sample
        self.lambda_prior = sc.random.normal(loc=self.mean_sample, scale=np.sqrt(self.variance_sample), size=1)[0]
        self.r_prior = sc.random.gamma(shape=1, scale=1. / self.variance_sample, size=1)[0]
        self.w_prior = sc.random.gamma(shape=1, scale=self.variance_sample, size=1)[0]
        self.Z = np.random.randint(0, K, size=self.M)

        means = []
        S = []
        for k in range(len(np.unique(self.Z))):
            Xk = self.omegas[np.where(self.Z == k)[0]]

            meank = Xk.mean()
            sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)
            S.append(1. / sigmak)
            means.append(meank)

        self.means = np.array(means)
        self.S = np.array(S)

    def initialize_multidimension(self, K):
        self.D = self.X.shape[1]
        self.omegas = np.random.rand(self.M, self.D)
        self.beta = np.random.rand(2 * self.M + self.bias, self.D)

        # self.mean_sample = self.omegas.mean(axis=0)
        self.mean_sample = np.zeros(self.D)

        self.variance_sample = 999999. * np.identity(self.D)
        self.inverse_variance_sample = np.linalg.inv(self.variance_sample)
        N = len(self.omegas)
        self.lambda_prior = sc.random.multivariate_normal(mean=self.mean_sample, cov=self.variance_sample, size=1)[
            0]
        self.r_prior = sc.stats.wishart.rvs(N - 1, self.inverse_variance_sample, size=1)
        self.w_prior = sc.stats.wishart.rvs(N - 1, self.variance_sample, size=1)

        kmeans = KMeans(n_clusters=K).fit(self.omegas)
        self.Z = np.array(kmeans.labels_)
        del kmeans

        means = []
        S = []
        for k in range(len(np.unique(self.Z))):
            Xk = self.omegas[np.where(self.Z == k)[0]]
            meank = Xk.mean(axis=0)
            if len(Xk) > 1:
                S.append(np.linalg.inv(np.cov(Xk.T)))
            else:
                S.append(np.identity(self.D) * 1. / 0.0001)
            means.append(meank)
            del meank, Xk

        self.means = np.array(means)
        self.S = np.array(S)

    @staticmethod
    def __sample_scaled_inverse_chi_square(scale, shape):
        """
        :param scale: Scale of inverse chi square
        :param shape: Shape of inverse chi square
        :return: A sample from a scaled inverse chi square
        """
        return sc.stats.invgamma.rvs(a=scale / 2., scale=scale * shape / 2., size=1)[0]

    @staticmethod
    def __multivariate_t_rvs(m, S, df=np.inf, n=1):
        """
        generate random variables of multivariate t distribution
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
        """
        m = np.asarray(m)
        d = len(m)
        if df == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        return m + z / np.sqrt(x)[:, None]  # same output format as random.multivariate_normal

    @staticmethod
    def __log_multivariate_t_pdf(X, m, S, df):
        p = len(m)
        up = sc.special.gammaln((df + p) / 2.)
        down = sc.special.gammaln(df / 2.) + p / 2. * (np.log(df) + np.log(np.pi)) + 1. / 2 * np.linalg.slogdet(S)[1]

        result = -(df + p) / 2. * np.log(1 + 1. / df * ((X - m).T.dot(np.linalg.inv(S).dot(X - m))))

        return up - down + result

    def __sample_w_beta_priors(self):
        K = len(self.means)
        if self.oneDimension:
            shape_value = K * self.beta_prior + 1
            scale_value = (
                    1. / (K * self.beta_prior + 1) * (1. / self.variance_sample + self.beta_prior * np.sum(self.S)))
            if scale_value <= 0:
                print("error")
            self.w_prior = sc.random.gamma(shape=shape_value, scale=1. / scale_value, size=1)[0]

        else:
            shape_value = K * self.beta_prior + self.D
            scale_value = (K * self.beta_prior + self.D) * np.linalg.inv(
                (self.inverse_variance_sample + self.beta_prior * np.sum(self.S, axis=0)))
            self.w_prior = sc.stats.invwishart.rvs(shape_value, scale_value, size=1)
        del shape_value, scale_value, K

    def sampling_Z(self):
        # Complexity is O(K M)
        if self.oneDimension:
            self.__sampling_Z_1_D()
        else:
            self.__sampling_Z_moredimesions()

    def __sampling_Z_moredimesions(self):
        # Complexity is O(M)
        N = self.M
        order_sampling_Z = list(range(len(self.Z)))
        random.shuffle(order_sampling_Z)
        r_prior_inverse = np.linalg.inv(self.r_prior)
        w_inverse = np.linalg.inv(self.w_prior)
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
                    while True:
                        try:
                            prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                                self.omegas[i], mean=self.means[k].reshape(self.D, ), cov=inverseS[k])
                            break
                        except (np.linalg.LinAlgError, ValueError):
                            inverseS[k] += 0.01 * np.identity(self.D)
                            self.S[k] += np.linalg.inv(inverseS[k])

                else:
                    meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)
                    sk = sc.stats.wishart.rvs(self.beta_prior + self.D, w_inverse, size=1)

                    try:
                        prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.omegas[i], meank,
                                                                                                  cov=np.linalg.inv(
                                                                                                      sk))
                    except np.linalg.LinAlgError:
                        sk += 0.01 * np.identity(self.D)
                        prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.omegas[i], meank,
                                                                                                  cov=np.linalg.inv(
                                                                                                      sk))

                    new_means_precision[k] = [meank, sk]

                probability_of_belongs_class.append(prob)
                del Nk, prob

            meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)

            sk = sc.stats.wishart.rvs(self.beta_prior + self.D, w_inverse, size=1)

            try:
                prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.omegas[i], meank,
                                                                                          cov=np.linalg.inv(sk))
            except (ValueError, np.linalg.LinAlgError):
                sk += 0.001 * np.identity(self.D)
                prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.omegas[i], meank,
                                                                                          cov=np.linalg.inv(sk))

            new_means_precision[K] = [meank, sk]

            probability_of_belongs_class.append(prob)

            del prob

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
                            self.means = np.concatenate((self.means.reshape(self.means.shape[0], self.D),
                                                         new_means_precision[k][0].reshape(1, self.D)), axis=0)
                            self.S = np.concatenate((self.S, new_means_precision[k][1].reshape(1, self.D, self.D)))
                            inverseS.append(np.linalg.inv(new_means_precision[k][1]))
                    self.Z[i] = k
                    del actualProb, k, new_means_precision
                    break
            del probability_of_belongs_class, sk, meank, u
        del i

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
            del k
        del inverseS, r_prior_inverse, w_inverse, order_sampling_Z, N, K

    def __sampling_Z_1_D(self):
        # Complexity is O(M)
        N = self.M
        order_sampling_Z = list(range(len(self.Z)))
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
                    prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (
                            sk * (self.omegas[i] - meank) ** 2) / 2.

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
            del probability_of_belongs_class, new_means_precision

        if len(self.means) != len(np.unique(self.Z)):
            k = 0
            while k < len(self.means):
                if len(np.where(self.Z == k)[0]) <= 0:
                    self.means = np.delete(self.means, k)
                    self.S = np.delete(self.S, k)
                    self.Z[np.where(self.Z > k)[0]] -= 1
                else:
                    k += 1
        del order_sampling_Z

    def sampling_mu_sigma(self):
        # Complexity O(K)
        if self.oneDimension:
            self.__sampling_mu_sigma_1_D()
        else:
            self.__sampling_mu_sigma_moredimension()

    def __sampling_mu_sigma_moredimension(self):
        K = len(self.means)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.omegas[np.where(self.Z == k)]
            meank = Xk.mean(axis=0)
            sk = self.S[k]
            Nk = len(Xk)

            try:
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior)
            except np.linalg.LinAlgError:
                # If is singular we add some value
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior + 0.001 * np.identity(self.D))
            mean_value = (Nk * meank.dot(sk) + self.lambda_prior.dot(self.r_prior)).dot(sigma_value).reshape(
                self.D, )

            newmeank = sc.stats.multivariate_normal.rvs(mean=mean_value, cov=sigma_value, size=1)
            del sigma_value, mean_value
            new_means.append(newmeank)
            # sampling sk
            shape_parameter = Nk - 1 + self.beta_prior + self.D
            S = np.zeros_like(self.w_prior)

            for x in Xk:
                S += (x - meank).reshape(self.D, 1).dot((x - meank).reshape(1, self.D))

            scale_parameter = np.linalg.inv((S + np.linalg.inv(self.w_prior * self.beta_prior)))
            try:
                newSk = sc.stats.wishart.rvs(shape_parameter, scale_parameter, 1)
                new_S.append(newSk)
            except np.linalg.LinAlgError:  # If is error we sample from a inverse wishart distribution and invert the matrix
                scale_parameter = S + np.linalg.inv(self.w_prior * self.beta_prior)
                try:
                    newSk = np.linalg.inv(sc.stats.invwishart.rvs(shape_parameter, scale_parameter))
                except np.linalg.LinAlgError:
                    scale_parameter += 0.001 * np.identity(self.D)
                    newSk = np.linalg.inv(sc.stats.invwishart.rvs(shape_parameter, scale_parameter))
                new_S.append(newSk)
            del S, newSk, scale_parameter, shape_parameter, newmeank, Xk, Nk, sk, x, meank

        self.means = np.array(new_means)
        self.S = np.array(new_S)
        del K, new_means, new_S, k

    def __sampling_mu_sigma_1_D(self):
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
            scale_paremeter = (1. / (self.beta_prior + Nk)) * (
                    self.w_prior * self.beta_prior + np.sum((Xk - meank) ** 2))
            sk = 1. / self.__sample_scaled_inverse_chi_square(shape_parameter, scale_paremeter)
            new_S.append(sk)
            del Xk

        self.means = np.array(new_means)
        self.S = np.array(new_S)
        del new_S, new_means

    def sample_beta_sigma(self, omegas=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if omegas is None:
            omegas = self.omegas
        if miu0 is None:
            Phi_x = self.matrix_phi(omegas)
            if self.bias > 0:
                b = np.ones((self.N, self.M * 2 + 1))
                b[:, 1:] = Phi_x
                Phi_x = b

            an = a0 + self.N / 2.
            invLambda0 = alpha_prior * np.identity(self.M * 2 + self.bias)
            invLambdan = Phi_x.T.dot(Phi_x) + invLambda0
            Lamdan = np.linalg.inv(invLambdan)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(invLambdan).dot(miun))
        else:
            miu0 = np.zeros(self.M * 2)
            Phi_x = self.matrix_phi(omegas)
            an = a0 + self.N / 2.
            invLambda0 = alpha_prior * np.identity(self.M * 2 + self.bias)
            invLambdan = Phi_x.T.dot(Phi_x) + invLambda0
            Lamdan = np.linalg.inv(invLambdan)
            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(invLambda0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLambda0).dot(miu0) - miun.T.dot(
                invLambdan).dot(miun))

        self.sigma_e = sc.stats.invgamma.rvs(a=an, scale=bn, size=1)[0]
        mean = miun
        Sigma = an / bn * Lamdan
        df = 2 * an
        self.beta = self.__multivariate_t_rvs(mean, Sigma, df)[0]

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
            del betahat, variance, Vbeta, Rinv, X, k, N, Q, R
            return betaProposal, sigma
        except:
            print("stop")
            # Checar cual de las dos
            # return betahat, sigma
            del betahat, variance, Rinv, X, k, N, Q, R
            return Vbeta, sigma

    def get_Phi_X(self, X, omegas):
        return self.matrix_phi_with_X(omegas, X)

    def predict(self, X):
        if self.oneDimension:  # If is one dimension
            self.sample_beta_sigma()
            Y = []
            Phi_X = self.get_Phi_X(X, self.omegas)
            for mean in Phi_X.dot(self.beta):
                Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        else:
            self.sample_beta_sigma()
            Phi_X = self.get_Phi_X(X, self.omegas)

            Y = []
            for mean in Phi_X.dot(self.beta):
                Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        return np.array(Y)

    def predict_new_X_beta_omega(self, X):
        self.sample_beta_sigma()
        Phi_X = self.get_Phi_X(X, self.omegas)

        Y = []
        for mean in Phi_X.dot(self.beta):
            Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        return np.array(Y)

    def predict_new_X(self, X, omegas=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if omegas is None:
            omegas = self.omegas
        if miu0 is None:
            Phi_x = self.matrix_phi(omegas)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)

            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

            # miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, omegas)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
            df = 2 * an
        else:

            Phi_x = self.matrix_phi(omegas)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)

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

            shape_value_r = K + 1 + self.D
            sum_result = np.zeros_like(variance_value_lambda)
            for row in self.means - self.lambda_prior:
                sum_result += row.reshape(self.D, 1).dot(row.reshape(1, self.D))
            scale_value_r = 1. / (K + 1) * np.linalg.inv(self.variance_sample + sum_result)
            del row, sum_result

            self.r_prior = sc.stats.wishart.rvs(shape_value_r, scale_value_r, size=1)
        del mean_value_lambda, scale_value_r, shape_value_r, variance_value_lambda, K

    def predict_new_X_mean(self, X, omegas=None, alpha_prior=0.000001, miu0=None):
        if omegas is None:
            omegas = self.omegas

        if miu0 is None:
            Phi_x = self.matrix_phi(omegas)
            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miu0 = np.zeros(self.M * 2 + 1)
            Phi_x = self.matrix_phi(omegas)
            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))

        del Phi_x, Lamda0, Lamdan
        Phi_x_new = self.get_Phi_X(X, omegas)
        mean = Phi_x_new.dot(miun)
        del Phi_x_new, miun
        return mean

    def return_mean_variance_learning(self, X, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if miu0 is None:
            Phi_x = self.matrix_phi(omegas)
            an = a0 + self.N / 2.
            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))
            Phi_x_new = self.get_Phi_X(X, omegas)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
        else:
            miu0 = np.zeros(self.M * 2 + 1)
            Phi_x = self.matrix_phi(omegas)
            an = a0 + self.N / 2.
            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, omegas)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
        del Phi_x, an, a0, omegas, X, b0, alpha_prior, miu0, Lamda0, Lamdan, miun, bn, Phi_x_new
        return mean, Sigma

    @staticmethod
    def phi_xi(x, w):
        argument = w.dot(x).T
        cons = 1. / np.sqrt(len(w))
        del w, x
        res = cons * np.concatenate((np.cos(argument), np.sin(argument)))
        del argument
        return res

    def matrix_phi_with_X(self, omegas, X):
        if self.D == 1:
            X = X.reshape(len(X), 1)
            argument = X.dot(omegas.reshape(1, len(omegas)))
        else:
            argument = X.dot(omegas.T)

        cons = 1. / np.sqrt(len(omegas))
        del X, omegas

        try:
            res = cons * np.column_stack((np.cos(argument), np.sin(argument)))
            del argument, cons
            if self.bias > 0:
                b = np.ones((self.N, self.M * 2 + self.bias))
                b[:, 1:] = res
                res = b
                del b
            return res
        except MemoryError:
            gc.collect()
            cos_argument = np.cos(argument)
            gc.collect()
            sin_argument = np.sin(argument)
            del argument
            gc.collect()
            res = cons * np.column_stack((cos_argument, sin_argument))
            del cos_argument, sin_argument, cons
            return res

        # return np.column_stack((np.tile(1, len(X)),
        # 1. / np.sqrt(len(omegas)) * np.column_stack((np.cos(argument), np.sin(argument)))))

    def matrix_phi(self, omegas):
        if self.D == 1:
            argument = self.X.reshape(self.N, 1).dot(omegas.reshape(1, len(omegas)))  # complexity O(N d M)
        else:
            argument = self.X.dot(omegas.T)  # complexity O(N d M)
        cons = 1. / np.sqrt(len(omegas))
        del omegas
        Phi_X = np.array(cons * np.column_stack((np.cos(argument), np.sin(argument))), float)
        del argument, cons
        if self.bias > 0:
            b = np.ones((self.N, self.M * 2 + self.bias))
            b[:, 1:] = Phi_X
            Phi_X = b
            del b

        return Phi_X

    def f(self, omegas):
        Phi_x = self.matrix_phi(omegas)
        del omegas
        means = np.array(Phi_x).dot(self.beta)
        del Phi_x
        Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
        del means
        return Y

    def get_beta_sigma_with_given_omega(self, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        # sample
        if miu0 is None:
            miu0 = np.zeros(self.M * 2 + 1)
        Phi_x = self.matrix_phi(omegas)
        an = a0 + self.N / 2.

        Lamda0 = alpha_prior * np.identity(self.M * 2 + self.bias)

        Lamdan = Phi_x.T.dot(Phi_x) + Lamda0

        miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))

        bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(Lamda0).dot(miu0) - miun.T.dot(Lamdan).dot(miun))
        sigma_e = sc.stats.invgamma(an, bn)

        beta_sampled = self.__multivariate_t_rvs(miun, bn / an * Lamdan, 2 * an)

        return beta_sampled, sigma_e

    def get_omegas(self):
        return self.omegas

    def get_beta_sigma(self):
        return self.beta, self.sigma_e

    def sample_priors(self):
        self.mean_sample = self.omegas.mean(axis=0)
        if self.oneDimension:
            self.variance_sample = float(np.mean(abs(self.omegas - self.mean_sample) ** 2))
            self.inverse_variance_sample = 1. / self.variance_sample
        else:
            self.variance_sample = np.cov(self.omegas.T)
            self.inverse_variance_sample = np.linalg.inv(self.variance_sample)
        self.__sample_w_beta_priors()
        self.__sample_lambda_r_priors()

    def get_pik(self):
        K = len(self.means)
        N = float(len(self.omegas))
        pik = np.zeros(K)
        for k in range(K):
            pik[k] = len(np.where(self.Z == k)[0]) / N

        return pik

    def get_covariance_matrix(self):
        if self.oneDimension:
            return 1. / self.S

        covarianceMatrices = []
        for cov in self.S:
            cov = np.linalg.inv(cov)
            covarianceMatrices.append(cov)
        return np.array(covarianceMatrices)

    def return_ROC(self, x_test, y_test):
        y_predict_proba = self.pr


class BankLocallyStationary:
    def __init__(self, X, Y, M1, M2, bias=0):
        self.X = X
        if 0 < bias < 1:
            bias = round(bias)
        elif bias > 1:
            bias = 1
        elif bias < 0:
            bias = 0

        self.bias = bias
        if len(Y.shape) == 1:
            self.Y = Y
        else:
            self.Y = Y.reshape(len(Y), )

        self.M1 = M1
        self.M2 = M2
        self.constant1 = 1. / np.sqrt(M1)
        self.constant2 = 1. / np.sqrt(M2)
        self.N = len(Y)
        self.oneDimension = (len(X.shape) == 1)

        while True:  # Some times is lunch an exception in the initialization
            try:
                self.k_stationary = BaseBank(X, Y, M1, bias)
            except np.linalg.LinAlgError:
                continue
            break

        while True:  # Some times is lunch an exception in the initialization
            try:
                self.k_positive = BaseBank(X, Y, M2, bias)
            except np.linalg.LinAlgError:
                continue
            break

        self.D = self.k_positive.D

        if self.k_stationary.oneDimension:
            self.frequencies = np.zeros(M1 * M2)
            self.beta = np.random.rand(2 * self.M1 * self.M2 + self.bias)
        else:
            self.frequencies = np.zeros((M1 * M2, self.D))
            self.beta = np.random.rand(2 * self.M1 * self.M2 + self.bias, self.D)

        self.sigma_e = 999999.
        del X, Y, M1, M2, bias

    def update_frequences(self):
        for i in range(len(self.k_stationary.omegas)):
            self.frequencies[i * self.M2: (i + 1) * self.M2] = self.k_stationary.omegas[i] + self.k_positive.omegas / 2

    def update_omega(self, j):
        self.frequencies[j * self.M2: (j + 1) * self.M2] = self.k_stationary.omegas[j] + self.k_positive.omegas / 2

    def get_frequence(self, omegas, vs):
        frequences = np.zeros_like(self.frequencies)
        for i in range(len(omegas)):
            frequences[i * self.M2: (i + 1) * self.M2] = omegas[i] + vs / 2

        return frequences

    def get_frequence_minus(self, omegas=None, vs=None):
        if omegas is None:
            omegas = self.get_omegas()

        if vs is None:
            vs = self.get_vs()

        frequences = np.zeros_like(self.frequencies)
        for i in range(len(omegas)):
            frequences[i * self.M2: (i + 1) * self.M2] = omegas[i] - vs / 2

        return frequences

    def matrix_phi_with_X(self, frequences, X):
        if self.oneDimension:
            argument = X.dot(frequences.reshape(1, len(frequences)))
        else:
            argument = X.dot(frequences.T)
        try:
            Phi_X = 1. / np.sqrt(len(frequences)) * np.column_stack((np.cos(argument), np.sin(argument)))
            if self.bias > 0:
                b = np.ones((len(X), self.M1 * self.M2 * 2 + self.bias))
                b[:, 1:] = Phi_X
                Phi_X = b
            return Phi_X
        except MemoryError:
            gc.collect()
            cos_argument = np.cos(argument)
            gc.collect()
            sin_argument = np.sin(argument)
            gc.collect()
            Phi_X = 1. / np.sqrt(len(frequences)) * np.column_stack((cos_argument, sin_argument))
            if self.bias > 0:
                b = np.ones((len(X), self.M1 * self.M2 * 2 + self.bias))
                b[:, 1:] = Phi_X
                Phi_X = b
            return Phi_X

    @staticmethod
    def chunkIt(X, num):
        avg = len(X) / float(num)
        out = []
        last = 0.0

        while last < len(X):
            out.append(X[int(last):int(last + avg)])
            last += avg

        return out

    @staticmethod
    def __multivariate_t_rvs(m, S, df=np.inf, n=1):
        """
        generate random variables of multivariate t distribution
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
        """
        m = np.asarray(m)
        d = len(m)
        if df == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        return m + z / np.sqrt(x)[:, None]  # same output format as random.multivariate_normal

    def matrix_phi(self, frequences):
        if self.oneDimension:
            argument = self.X.reshape(self.N, 1).dot(frequences.reshape(1, len(frequences)))  # complexity O(N d M)
        else:
            argument = self.X.dot(frequences.T)  # complexity O(N d M)
        C = 1. / len(frequences)
        Phi_X = np.array(C * np.column_stack((np.cos(argument), np.sin(argument))), float)
        if self.bias > 0:
            b = np.ones((self.N, 1))
            Phi_X = np.c_[b, Phi_X]
        return Phi_X

    @staticmethod
    def __sample_scaled_inverse_chi_square(scale, shape):
        """
        :param scale: Scale of inverse chi square
        :param shape: Shape of inverse chi sqaure
        :return: A sample from a scaled inverse chi square
        """
        return sc.stats.invgamma.rvs(a=scale / 2., scale=scale * shape / 2., size=1)[0]

    @staticmethod
    def __log_multivariate_t_pdf(X, m, S, df):
        p = len(m)
        up = sc.special.gammaln((df + p) / 2.)
        down = sc.special.gammaln(df / 2.) + p / 2. * (np.log(df) + np.log(np.pi)) + 1. / 2 * np.linalg.slogdet(S)[1]

        result = -(df + p) / 2. * np.log(1 + 1. / df * ((X - m).T.dot(np.linalg.inv(S).dot(X - m))))

        return up - down + result

    def sampling_Z(self):
        # Complexity is O(K1 M1 + K2 * M2)
        self.k_stationary.sampling_Z()
        self.k_positive.sampling_Z()

    def sampling_mu_sigma(self):
        # Complexity O(K1 + K2)
        self.k_positive.sampling_mu_sigma()
        self.k_stationary.sampling_mu_sigma()

    def sample_beta_sigma(self, frequences=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequencies

        Phi_x = self.matrix_phi(frequences)
        an = a0 + self.N / 2.
        invLambda0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        invLambdan = Phi_x.T.dot(Phi_x) + invLambda0
        Lamdan = np.linalg.inv(invLambdan)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(invLambdan).dot(miun))
        else:
            miun = Lamdan.dot(invLambda0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLambda0).dot(miu0) - miun.T.dot(
                invLambdan).dot(miun))

        self.sigma_e = sc.stats.invgamma.rvs(a=an, scale=bn, size=1)[0]
        mean = miun
        Sigma = an / bn * Lamdan
        df = 2 * an
        self.beta = self.__multivariate_t_rvs(mean, Sigma, df)[0]

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

    def get_Phi_X(self, X, frequences):
        return self.matrix_phi_with_X(frequences, X)

    def get_phi_minus_X(self, X, frequences):
        return self.matrix_phi_with_X(frequences, X)

    def predict(self, X):
        if self.oneDimension:  # If is one dimension
            self.sample_beta_sigma()
            Y = []
            Phi_X = self.get_Phi_X(X, self.get_frequence_minus())
            for mean in Phi_X.dot(self.beta):
                Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        else:
            self.sample_beta_sigma()
            Phi_X = self.get_Phi_X(X, self.frequencies)

            Y = []
            for mean in Phi_X.dot(self.beta):
                Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        return np.array(Y)

    def predict_new_X_beta_omega(self, X):
        self.sample_beta_sigma()
        # Phi_X = self.get_Phi_X(X, self.frequences)
        Phi_X = self.get_Phi_X(X, self.frequencies)

        Y = []
        for mean in Phi_X.dot(self.beta):
            Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        return np.array(Y)

    def predict_new_X(self, X, frequences=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            # frequences = self.frequences
            frequences = self.get_frequence_minus()

        Phi_x = self.matrix_phi(frequences)
        an = a0 + self.N / 2.
        Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))
        else:
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

        Phi_x_new = self.get_Phi_X(X, frequences)
        mean = Phi_x_new.dot(miun)
        Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
        df = 2 * an
        return self.__multivariate_t_rvs(mean, Sigma, df, n=len(X))[0]

    def predict_new_X_mean(self, X, frequences=None, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequencies

        Phi_x = self.matrix_phi(frequences)
        Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))

        Phi_x_new = self.get_Phi_X(X, frequences)
        mean = Phi_x_new.dot(miun)
        return mean

    def return_mean_variance_learning(self, X, frequences, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        Phi_x = self.matrix_phi(frequences)
        an = a0 + self.N / 2.
        Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))
        else:
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

        Phi_x = self.get_Phi_X(X, frequences)
        mean = Phi_x.dot(miun)
        Sigma = an / bn * (np.identity(len(X)) + Phi_x.dot(Lamdan).dot(Phi_x.T))
        return mean, Sigma

    def f(self, frequences, beta_local):
        Phi_x = self.matrix_phi(frequences)

        means = np.array(Phi_x).dot(beta_local)
        Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
        return Y

    def get_omegas(self):
        return self.k_stationary.omegas

    def get_vs(self):
        return self.k_positive.omegas

    def get_beta_sigma(self):
        return self.beta, self.sigma_e

    def sample_priors(self):
        self.k_positive.sample_priors()
        self.k_positive.sample_priors()

    def get_pik_omega(self):
        return self.k_stationary.get_pik()

    def get_covariance_matrix_omega(self):
        return self.k_stationary.get_covariance_matrix()

    def get_pik_vs(self):
        return self.k_positive.get_pik()

    def get_covariance_matrix_vs(self):
        return self.k_positive.get_covariance_matrix()


'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random
import scipy as sc
import scipy.special
import numpy as np
from scipy.stats import beta
from sklearn.cluster import KMeans


class bank:
    def __init__(self, X, Y, M1, M2):
        self.X = X
        self.Y = Y
        self.M1 = M1
        self.M2 = M2
        self.N = len(Y)
        self.oneDimension = (len(X.shape) == 1)
        K1 = np.random.randint(2, 14, size=1)[0]
        K2 = np.random.randint(2, 14, size=1)[0]
        if self.oneDimension:
            self.D = 1
            self.omegas = np.random.rand(M1)
            self.vs = np.random.rand(M2)
            self.beta = np.random.rand(2 * self.M1 * self.M2)
            self.mean_sample = self.omegas.mean(axis=0)

            self.variance_sample = 99999.
            self.inverse_variance_sample = 1. / self.variance_sample
            self.lambda_prior = sc.random.normal(loc=self.mean_sample, scale=np.sqrt(self.variance_sample), size=1)[0]
            self.r_prior = sc.random.gamma(shape=1, scale=1. / self.variance_sample, size=1)[0]
            self.w_prior = sc.random.gamma(shape=1, scale=self.variance_sample, size=1)[0]
            self.Z1 = np.random.randint(0, K1, size=self.M1)

            self.means_omega = []
            self.S_omega = []
            for k in range(len(np.unique(self.Z1))):
                Xk = self.omegas[np.where(self.Z1 == k)[0]]
                meank = Xk.mean()
                sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)
                self.S_omega.append(1. / sigmak)
                self.means_omega.append(meank)

            self.means_omega = np.array(self.means_omega)
            self.S_omega = np.array(self.S_omega)
            self.Z2 = np.random.randint(0, K2, size=self.M1)
            self.means_v = []
            self.S_v = []
            for k in range(len(np.unique(self.Z2))):
                Xk = self.omegas[np.where(self.Z2 == k)[0]]
                meank = Xk.mean()
                sigmak = np.mean(abs(Xk - Xk.mean()) ** 2)
                self.S_v.append(1. / sigmak)
                self.means_v.append(meank)

            self.means_v = np.array(self.means_v)
            self.S_v = np.array(self.S_v)
            self.frequences = np.zeros(self.M1 * self.M2)
        else:
            self.D = X.shape[1]
            self.omegas = np.random.rand(M1, self.D)
            self.vs = np.random.rand(M2, self.D)
            self.beta = np.random.rand(2 * self.M1*self.M2, self.D)

            self.mean_sample = self.omegas.mean(axis=0)

            self.variance_sample = 999999. * np.identity(self.D)
            self.inverse_variance_sample = np.linalg.inv(self.variance_sample)
            N = len(self.omegas)
            self.lambda_prior = sc.random.multivariate_normal(mean=self.mean_sample, cov=self.variance_sample, size=1)[0]
            self.r_prior = sc.stats.wishart.rvs(N - 1, self.inverse_variance_sample, size=1)
            self.w_prior = sc.stats.wishart.rvs(N - 1, self.variance_sample, size=1)

            kmeans = KMeans(n_clusters=K1).fit(self.omegas)
            self.Z1 = np.array(kmeans.labels_)

            self.means_omega = []
            self.S_omega = []
            for k in range(len(np.unique(self.Z1))):
                Xk = self.omegas[np.where(self.Z1 == k)[0]]
                meank = Xk.mean(axis=0)
                self.S_omega.append(np.linalg.inv(np.cov(Xk.T)))
                self.means_omega.append(meank)

            self.means_omega = np.array(self.means_omega)
            self.S_omega = np.array(self.S_omega)

            kmeans = KMeans(n_clusters=K2).fit(self.vs)
            self.Z2 = np.array(kmeans.labels_)

            self.means_v = []
            self.S_v = []
            for k in range(len(np.unique(self.Z2))):
                Xk = self.omegas[np.where(self.Z2 == k)[0]]
                meank = Xk.mean(axis=0)
                self.S_v.append(np.linalg.inv(np.cov(Xk.T)))
                self.means_v.append(meank)

            self.means_v = np.array(self.means_v)
            self.S_v = np.array(self.S_v)

            self.frequences = np.zeros((self.M1 * self.M2, self.D))

        self.alpha = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.beta_prior = 1. / sc.random.gamma(shape=1, scale=1, size=1)[0]
        self.sigma_e = 9999999
        self.__updatefrequences()

    def __updatefrequences(self):
        if self.D == 1:
            self.frequences = np.array([])
            for v in self.vs:
                vec_freq = v/2. + self.omegas
                self.frequences = np.append(self.frequences, vec_freq)

        else:
            for i in range(self.M2):
                vec_freq = self.vs[i]/2. + self.omegas
                self.frequences[i * self.M1: (i + 1) * self.M1] = vec_freq


    def getfrequences_new_omega(self, new_w, i):
        frequences = self.frequences.copy()
        for j in range(self.M2):
            frequences[j * self.M1 + i] = self.vs[i]/2. + new_w

        return frequences

    def getfrequences_new_vs(self, new_v, i):
        frequences = self.frequences.copy()
        vec_freq = new_v/2. + self.omegas
        frequences[i * self.M1: (i + 1) * self.M1] = vec_freq
        return frequences

    @staticmethod
    def __sample_scaled_inverse_chi_square(scale, shape):
        '''
        :param scale: Scale of inverse chi square
        :param shape: Shape of inverse chi sqaure
        :return: A sample from a scaled inverse chi square
        '''
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

    @staticmethod
    def __log_multivariate_t_pdf(X, m, S, df):
        p = len(m)
        up = sc.special.gammaln((df+p)/2.)
        down = sc.special.gammaln(df/2.) + p/2. *(np.log(df) + np.log(np.pi)) + 1./2 * np.linalg.slogdet(S)[1]
        result = -(df + p)/2.*np.log(1 + 1./df*((X - m).T.dot(np.linalg.inv(S).dot(X-m))))
        return up - down + result

    def __sample_w_beta_priors(self):
        K = len(self.means_omega)
        if self.oneDimension:
            shape_value = K * self.beta_prior + 1
            scale_value = (1. / (K * self.beta_prior + 1) * (1. / self.variance_sample + self.beta_prior * np.sum(self.S_omega)))
            if scale_value <= 0:
                print("error")
            self.w_prior = sc.random.gamma(shape=shape_value, scale=1./scale_value, size=1)[0]
        else:
            shape_value = K * self.beta_prior + self.D
            scale_value = (K * self.beta_prior + self.D) * np.linalg.inv((self.inverse_variance_sample + self.beta_prior * np.sum(self.S_omega, axis=0)))
            self.w_prior = sc.stats.invwishart.rvs(shape_value, scale_value, size=1)

    def sampling_Z(self):
        if self.oneDimension:
            self.__sampling_Z1_one_dimension()
            self.__sampling_Z2_one_dimension()
        else:
            self.__sampling_Z1_moredimension()
            self.__sampling_Z2_moredimension()

    def __sampling_Z2_one_dimension(self):
        N = self.M2
        order_sampling_Z = range(len(self.Z2))
        random.shuffle(order_sampling_Z)
        # shuffle the index to more fastest converge
        for i in order_sampling_Z:
            K = len(self.means_v)
            new_means_precision = {}
            probability_of_belongs_class = []
            for k in range(K):
                Nk = len(np.where(self.Z2 == k)[0])
                if self.Z2[i] == k:
                    Nk -= 1

                if Nk > 0:
                    prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(self.S_v[k]) - (
                            self.S_v[k] * (self.vs[i] - self.means_v[k]) ** 2) / 2.
                else:
                    meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
                    sk = sc.random.gamma(shape=self.beta_prior, scale=1. / self.w_prior, size=1)[0]

                    new_means_precision[k] = [meank, sk]
                    prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (sk * (self.vs[i] - meank) ** 2) / 2.

                probability_of_belongs_class.append(prob)
            meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
            sk = sc.random.gamma(shape=self.beta_prior, scale=1. / self.w_prior, size=1)[0]

            new_means_precision[K] = [meank, sk]
            prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (sk * (self.vs[i] - meank) ** 2) / 2.
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
                            self.means_v[k] = new_means_precision[k][0]
                            self.S_v[k] = new_means_precision[k][1]
                        else:
                            self.means_v = np.append(self.means_v, new_means_precision[k][0])
                            self.S_v = np.append(self.S_v, new_means_precision[k][1])
                    self.Z2[i] = k
                    break
        if len(self.means_v) != len(np.unique(self.Z2)):
            k = 0
            while k < len(self.means_v):
                if len(np.where(self.Z2 == k)[0]) <= 0:
                    self.means_v = np.delete(self.means_v, k)
                    self.S_v = np.delete(self.S_v, k)
                    self.Z2[np.where(self.Z2 > k)[0]] -= 1
                else:
                    k += 1

    def __sampling_Z2_moredimension(self):
        N = self.M2
        order_sampling_Z = range(len(self.Z2))
        random.shuffle(order_sampling_Z)  # shuffle the index to more fastest converge
        r_prior_inverse = np.linalg.inv(self.r_prior)
        w_inverse = np.linalg.inv(self.w_prior)
        inverseS = []
        for k in range(len(self.means_v)):
            try:
                inverseS.append(np.linalg.inv(self.S_v[k]))
            except np.linalg.LinAlgError:
                inverseS.append(np.linalg.inv(self.S_v[k] + 0.0001 * np.identity(self.D)))

        for i in order_sampling_Z:
            K = len(self.means_v)
            new_means_precision = {}
            probability_of_belongs_class = []
            for k in range(K):
                Nk = len(np.where(self.Z2 == k)[0])
                if self.Z2[i] == k:
                    Nk -= 1
                if Nk > 0:
                    try:
                        prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                            self.vs[i], mean=self.means_v[k].reshape(self.D, ), cov=inverseS[k])
                    except (np.linalg.LinAlgError, ValueError):
                        self.S_omega[k] += 0.01 * np.identity(self.D)
                        inverseS[k] = np.linalg.inv(self.S_v[k])
                        prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                            self.vs[i], mean=self.means_v[k].reshape(self.D, ), cov=inverseS[k])
                else:
                    meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)
                    sk = sc.stats.wishart.rvs(self.beta_prior + self.D, w_inverse, size=1)

                    try:
                        prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.vs[i], meank,
                                                                                                  cov=np.linalg.inv(
                                                                                                      sk))
                    except np.linalg.LinAlgError:
                        sk += 0.01 * np.identity(self.D)
                        prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.vs[i], meank,
                                                                                                  cov=np.linalg.inv(
                                                                                                      sk))
                    new_means_precision[k] = [meank, sk]

                probability_of_belongs_class.append(prob)
            meank = sc.stats.multivariate_normal.rvs(self.lambda_prior, r_prior_inverse, size=1)

            sk = sc.stats.wishart.rvs(self.beta_prior + self.D, w_inverse, size=1)

            try:
                prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.vs[i], meank,
                                                                                          cov=np.linalg.inv(sk))
            except (ValueError, np.linalg.LinAlgError):
                sk += 0.001 * np.identity(self.D)
                prob = - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(self.vs[i], meank,
                                                                                          cov=np.linalg.inv(sk))
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
                                self.means_v[k] = new_means_precision[k][0].reshape(self.D, 1)
                            except ValueError:
                                self.means_v[k] = new_means_precision[k][0].reshape(self.D, )
                            self.S_v[k] = new_means_precision[k][1]
                            inverseS[k] = np.linalg.inv(self.S_v[k])
                        else:
                            self.means_v = np.concatenate(
                                (self.means_v.reshape(self.means_v.shape[0], self.D, 1),
                                 new_means_precision[k][0].reshape(1, self.D, 1)), axis=0)
                            self.S_v = np.concatenate((self.S_v, new_means_precision[k][1].reshape(1, 2, 2)))
                            inverseS.append(np.linalg.inv(new_means_precision[k][1]))
                    self.Z2[i] = k
                    break

        if len(self.means_v) != len(np.unique(self.Z2)):
            k = 0
            while k < len(self.means_v):
                if len(np.where(self.Z2 == k)[0]) <= 0:
                    self.means_v = np.delete(self.means_v, k, axis=0)
                    self.S_v = np.delete(self.S_v, k, axis=0)
                    del inverseS[k]
                    self.Z2[np.where(self.Z2 > k)[0]] -= 1
                else:
                    k += 1

    def __sampling_Z1_one_dimension(self):
        N = self.M1
        order_sampling_Z = range(len(self.Z1))
        random.shuffle(order_sampling_Z)
        # shuffle the index to more fastest converge
        for i in order_sampling_Z:
            K = len(self.means_omega)
            new_means_precision = {}
            probability_of_belongs_class = []
            for k in range(K):
                Nk = len(np.where(self.Z1 == k)[0])
                if self.Z1[i] == k:
                    Nk -= 1

                if Nk > 0:
                    prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(self.S_omega[k]) - (
                            self.S_omega[k] * (self.omegas[i] - self.means_omega[k]) ** 2) / 2.
                else:
                    meank = sc.random.normal(loc=self.lambda_prior, scale=np.sqrt(1. / self.r_prior), size=1)[0]
                    sk = sc.random.gamma(shape=self.beta_prior, scale=1. / self.w_prior, size=1)[0]

                    new_means_precision[k] = [meank, sk]
                    prob = - np.log(N - 1 + self.alpha) + 1. / 2 * np.log(sk) - (sk * (self.omegas[i] - meank) ** 2) / 2.

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
                            self.means_omega[k] = new_means_precision[k][0]
                            self.S_omega[k] = new_means_precision[k][1]
                        else:
                            self.means_omega = np.append(self.means_omega, new_means_precision[k][0])
                            self.S_omega = np.append(self.S_omega, new_means_precision[k][1])
                    self.Z1[i] = k
                    break
        if len(self.means_omega) != len(np.unique(self.Z1)):
            k = 0
            while k < len(self.means_omega):
                if len(np.where(self.Z1 == k)[0]) <= 0:
                    self.means_omega = np.delete(self.means_omega, k)
                    self.S_omega = np.delete(self.S_omega, k)
                    self.Z1[np.where(self.Z1 > k)[0]] -= 1
                else:
                    k += 1

    def __sampling_Z1_moredimension(self):
        N = self.M1
        order_sampling_Z = range(len(self.Z1))
        random.shuffle(order_sampling_Z)  # shuffle the index to more fastest converge
        r_prior_inverse = np.linalg.inv(self.r_prior)
        w_inverse = np.linalg.inv(self.w_prior)
        inverseS = []
        for k in range(len(self.means_omega)):
            try:
                inverseS.append(np.linalg.inv(self.S_omega[k]))
            except np.linalg.LinAlgError:
                inverseS.append(np.linalg.inv(self.S_omega[k] + 0.0001 * np.identity(self.D)))

        for i in order_sampling_Z:
            K = len(self.means_omega)
            new_means_precision = {}
            probability_of_belongs_class = []
            for k in range(K):
                Nk = len(np.where(self.Z1 == k)[0])
                if self.Z1[i] == k:
                    Nk -= 1
                if Nk > 0:
                    try:
                        prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                            self.omegas[i], mean=self.means_omega[k].reshape(self.D, ), cov=inverseS[k])
                    except (np.linalg.LinAlgError, ValueError):
                        self.S_omega[k] += 0.01 * np.identity(self.D)
                        inverseS[k] = np.linalg.inv(self.S_omega[k])
                        prob = np.log(float(Nk)) - np.log(N - 1 + self.alpha) + sc.stats.multivariate_normal.logpdf(
                            self.omegas[i], mean=self.means_omega[k].reshape(self.D, ), cov=inverseS[k])
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
                                self.means_omega[k] = new_means_precision[k][0].reshape(self.D, 1)
                            except ValueError:
                                self.means_omega[k] = new_means_precision[k][0].reshape(self.D, )
                            self.S_omega[k] = new_means_precision[k][1]
                            inverseS[k] = np.linalg.inv(self.S_omega[k])
                        else:
                            self.means_omega = np.concatenate(
                                (self.means_omega.reshape(self.means_omega.shape[0], self.D, 1),
                                 new_means_precision[k][0].reshape(1, self.D, 1)), axis=0)
                            self.S_omega = np.concatenate((self.S_omega, new_means_precision[k][1].reshape(1, 2, 2)))
                            inverseS.append(np.linalg.inv(new_means_precision[k][1]))
                    self.Z1[i] = k
                    break

        if len(self.means_omega) != len(np.unique(self.Z1)):
            k = 0
            while k < len(self.means_omega):
                if len(np.where(self.Z1 == k)[0]) <= 0:
                    self.means_omega = np.delete(self.means_omega, k, axis=0)
                    self.S_omega = np.delete(self.S_omega, k, axis=0)
                    del inverseS[k]
                    self.Z1[np.where(self.Z1 > k)[0]] -= 1
                else:
                    k += 1

    def sampling_mu_sigma_omega(self):
        if self.oneDimension:
            self.__sampling_mu_sigma_omega_1dimension()
        else:
            self.__sampling_mu_sigma_omega_moredimension()

    def __sampling_mu_sigma_omega_1dimension(self):
        K = len(self.means_omega)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.omegas[np.where(self.Z1 == k)]
            meank = Xk.mean()
            sk = self.S_omega[k]
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

        self.means_omega = np.array(new_means)
        self.S_omega = np.array(new_S)

    def __sampling_mu_sigma_omega_moredimension(self):
        K = len(self.means_omega)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.omegas[np.where(self.Z1 == k)]
            meank = Xk.mean(axis=0)
            sk = self.S_omega[k]
            Nk = len(Xk)
            try:
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior)
            except np.linalg.LinAlgError:
                # If is singular we add some value
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior + 0.001 * np.identity(self.D))
            mean_value = (Nk * meank.dot(sk) + self.lambda_prior.dot(self.r_prior)).dot(sigma_value).reshape(
                self.D, )

            newmeank = sc.stats.multivariate_normal.rvs(mean=mean_value, cov=sigma_value, size=1)

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
            except np.linalg.LinAlgError:
                newSk = sc.stats.wishart.rvs(shape_parameter,
                                             scale_parameter + 0.0001 * np.identity(len(scale_parameter)), 1)
                new_S.append(newSk)

        self.means_omega = np.array(new_means)
        self.S_omega = np.array(new_S)

    def sampling_mu_sigma_v(self):
        if self.oneDimension:
            self.__sampling_mu_sigma_v_1dimension()
        else:
            self.__sampling_mu_sigma_v_moredimension()

    def __sampling_mu_sigma_v_1dimension(self):
        K = len(self.means_v)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.vs[np.where(self.Z2 == k)]
            meank = Xk.mean()
            sk = self.S_v[k]
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

        self.means_v = np.array(new_means)
        self.S_v = np.array(new_S)

    def __sampling_mu_sigma_v_moredimension(self):
        K = len(self.means_v)
        new_means = []
        new_S = []

        for k in range(K):
            # sampling muk
            Xk = self.vs[np.where(self.Z1 == k)]
            meank = Xk.mean(axis=0)
            sk = self.S_v[k]
            Nk = len(Xk)
            try:
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior)
            except np.linalg.LinAlgError:
                # If is singular we add some value
                sigma_value = np.linalg.inv(Nk * sk + self.r_prior + 0.001 * np.identity(self.D))
            mean_value = (Nk * meank.dot(sk) + self.lambda_prior.dot(self.r_prior)).dot(sigma_value).reshape(
                self.D, )

            newmeank = sc.stats.multivariate_normal.rvs(mean=mean_value, cov=sigma_value, size=1)

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
            except np.linalg.LinAlgError:
                newSk = sc.stats.wishart.rvs(shape_parameter,
                                             scale_parameter + 0.0001 * np.identity(len(scale_parameter)), 1)
                new_S.append(newSk)

        self.means_v = np.array(new_means)
        self.S_v = np.array(new_S)


    def sample_beta_sigma(self, frequences=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequences

        if miu0 is None:
            Phi_x = self.matrix_phi(frequences)
            an = a0 + self.N/2.
            invLambda0 = alpha_prior*np.identity(self.M1 * self.M2*2)

            invLambdan = Phi_x.T.dot(Phi_x) + invLambda0
            Lamdan = np.linalg.inv(invLambdan)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(invLambdan).dot(miun))
        else:
            miu0 = np.zeros(self.M1 * self.M2 * 2)
            Phi_x = self.matrix_phi(frequences)
            an = a0 + self.N / 2.
            invLambda0 = alpha_prior * np.identity(self.M1 * self.M2 * 2)
            invLambdan = Phi_x.T.dot(Phi_x) + invLambda0
            Lamdan = np.linalg.inv(invLambdan)
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
        return self.matrix_phi_with_X(X, frequences)

    def predict_new_X_beta_omega(self, X):
        self.sample_beta_sigma()
        Phi_X = self.get_Phi_X(X, self.frequences)

        Y = []
        for mean in Phi_X.dot(self.beta):
            Y.append(np.random.normal(loc=mean, scale=self.sigma_e))
        return np.array(Y)

    def predict_new_X(self, X, frequences=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        if frequences is None:
            frequences = self.frequences

        if miu0 is None:
            Phi_x = self.matrix_phi(frequences)
            an = a0 + self.N/2.
            Lamda0 = 1./alpha_prior*np.identity(self.M1*self.M2*2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, frequences)
            mean = Phi_x_new.dot(miun)
            Sigma = an/bn*(np.identity(len(X))+ Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
            df = 2*an
        else:

            Phi_x = self.matrix_phi(frequences)
            an = a0 + self.N / 2.
            Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, frequences)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
            df = 2 * an
        return self.__multivariate_t_rvs(mean, Sigma, df, n=len(X))[0]

    def __sample_lambda_r_priors(self):
        K = len(self.means_omega)
        if self.oneDimension:
            mean_value_lambda = (self.mean_sample * (1. / self.variance_sample) + self.r_prior * np.sum(self.means_omega)) / (
                    1. / self.variance_sample + K * self.r_prior)
            variance_value_lambda = 1. / ((1. / self.variance_sample) + K * self.r_prior)
            self.lambda_prior = sc.random.normal(loc=mean_value_lambda, scale=np.sqrt(variance_value_lambda), size=1)[0]

            shape_value_r = K + 1
            scale_value_r = 1. / (1. / (K + 1) * (self.variance_sample + np.sum(self.means_omega - self.lambda_prior) ** 2))
            self.r_prior = sc.random.gamma(shape=shape_value_r, scale=scale_value_r, size=1)[0]
        else:
            variance_value_lambda = np.linalg.inv(self.inverse_variance_sample + K * self.r_prior)
            mean_value_lambda = (self.mean_sample.dot(self.inverse_variance_sample) + self.r_prior.dot(
                np.sum(self.means_omega, axis=0))).dot(variance_value_lambda)

            self.lambda_prior = sc.stats.multivariate_normal.rvs(mean=mean_value_lambda, cov=variance_value_lambda,
                                                                 size=1)

            shape_value_r = K + 1
            sum_result = np.zeros_like(variance_value_lambda)
            for row in self.means_omega - self.lambda_prior:
                sum_result += row.reshape(self.D, 1).dot(row.reshape(1, self.D))
            scale_value_r = 1. / (K + 1) * np.linalg.inv(self.variance_sample + sum_result)

            self.r_prior = sc.stats.wishart.rvs(shape_value_r, scale_value_r, size=1)

    def return_mean_variance_learning(self, X, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        if miu0 is None:
            Phi_x = self.matrix_phi()
            an = a0 + self.N/2.
            Lamda0 = 1./alpha_prior*np.identity(self.M1 * self.M2 *2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, self.frequences)
            mean = Phi_x_new.dot(miun)
            Sigma = an/bn*(np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
        else:

            Phi_x = self.matrix_phi()
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0))
            miun = Lamdan.dot(np.linalg.inv(Lamda0).dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(np.linalg.inv(Lamda0)).dot(miu0) - miun.T.dot(
                np.linalg.inv(Lamdan)).dot(miun))

            Phi_x_new = self.get_Phi_X(X, self.frequences)
            mean = Phi_x_new.dot(miun)
            Sigma = an / bn * (np.identity(len(X)) + Phi_x_new.dot(Lamdan).dot(Phi_x_new.T))
        return mean, Sigma

    @staticmethod
    def phi_xi(x, w):
        argument = w.dot(x).T

        return 1./np.sqrt(len(w)) * np.concatenate((np.cos(argument), np.sin(argument)))

    def matrix_phi_with_X(self, X, omegas = None):
        means = []
        if omegas == None:
            frequences = self.frequences
        else:
            frequences = omegas

        for x in X:
            means.append(self.phi_xi(x, frequences))
        return np.array(means)

    def matrix_phi(self, omegas = None):
        means = []
        if omegas == None:
            frequences = self.frequences
        else:
            frequences = omegas
        for x in self.X:
            means.append(self.phi_xi(x, frequences))
        return np.array(means)

    def f(self, omegas):
        Phi_x = self.matrix_phi(omegas)

        means = np.array(Phi_x).dot(self.beta)
        Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
        return Y

    def get_beta_sigma_with_given_frequences(self, frequences, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        #sample
        if miu0 is None:
            miu0 = np.zeros(self.M1 * self.M2 * 2)
        Phi_x = self.matrix_phi(frequences)
        an = a0 + self.N/2.
        Lamda0 = alpha_prior*np.identity(self.M1 * self.M2*2)
        Lamdan =  Phi_x.T.dot(Phi_x) + Lamda0
        miun = np.linalg.inv(Lamdan).dot(Lamda0.dot(miu0) + Phi_x.T.dot(self.Y))
        bn = b0 + 1./2*(self.Y.dot(self.Y) + miu0.T.dot(Lamda0).dot(miu0) - miun.T.dot(Lamdan).dot(miun))
        sigma_e = sc.stats.invgamma(an, bn)

        beta_sampled = self.__multivariate_t_rvs(miun, bn/an*Lamdan, 2*an)

        return beta_sampled, sigma_e

    def get_omegas(self):
        return self.omegas

    def get_vs(self):
        return self.vs

    def get_frequences(self):
        return self.frequences

    def get_beta_sigma(self):
        return self.beta, self.sigma_e

    def __sample_priors(self):
        self.__sample_w_beta_priors()
        self.__sample_lambda_r_priors()
        self.mean_sample = self.omegas.mean(axis=0)
        self.variance_sample = float(np.mean(abs(self.omegas - self.mean_sample) ** 2))
        self.inverse_variance_sample = 1. / self.variance_sample

    def get_pik(self):
        K = len(self.means_omega)
        N = float(len(self.omegas))
        pik = np.zeros(K)
        for k in range(K):
            pik[k] = len(np.where(self.Z1 == k)[0]) / N

        return pik

    def get_covariance_matrix(self):
        covarianceMatrices = []
        for cov in self.S_omega:
            cov = np.linalg.inv(cov)
            covarianceMatrices.append(cov)
        return np.array(covarianceMatrices)

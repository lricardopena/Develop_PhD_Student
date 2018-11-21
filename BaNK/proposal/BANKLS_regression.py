'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random
from scipy.stats import beta
import numpy as np
import scipy as sc
import scipy.special
from sklearn.cluster import KMeans
from BaNK_locally import bank


class bank_regression(bank):
    def __compute_log_model_evidence(self, frequences, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        '''
        :param frequences: the matrix with all w and v in v/2 + w for all v and w
        :param a0: loc prior for sigma value
        :param b0: scale prior for sigma value
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with the frequences corresponding
        '''
        if miu0 is None:
            Phi_x = self.__matrix_phi(frequences)
            an = a0 + self.N/2.
            invLamba0 = alpha_prior*np.identity(self.M1 * self.M2 * 2)

            invLamban = Phi_x.T.dot(Phi_x) + invLamba0
            Lamdan = np.linalg.inv(invLamban)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(invLamban).dot(miun))
        else:
            Phi_x = self.__matrix_phi(frequences)
            an = a0 + self.N / 2.

            Lamda0 = 1. / alpha_prior * np.identity(self.M1 * self.M2 * 2 + 1)
            invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2)

            invLambdan = Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0)
            Lamdan = np.linalg.inv(invLambdan)

            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLamba0).dot(miu0) - miun.T.dot(
                invLambdan).dot(miun))

        result_log = 1./2*np.linalg.slogdet(Lamdan)[1] + a0*np.log(b0) + sc.special.gammaln(an)
        result_log +=- an * np.log(bn) - sc.special.gammaln(a0) - (self.M1 * self.M2 * 2 + 1) / 2. * np.log(1. / alpha_prior)
        return result_log

    def __sample_omega(self):
        actual__log_error = self.__compute_log_model_evidence(self.frequences)
        for j in range(self.M1):
            Zj = self.Z1[j]

            if self.oneDimension:
                w_proposal = sc.random.normal(self.means_omega[Zj], np.sqrt(1. / self.S_omega[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.means_omega[Zj], np.linalg.inv(self.S_omega[Zj]), size=1)[0]

            new_frequencies = bank.getfrequences_new_omega(self, w_proposal, j)
            new_log_error = self.__compute_log_model_evidence(new_frequencies)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.omegas[j] = w_proposal
                self.frequences = new_frequencies
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Acept with certain probability
                    actual__log_error = new_log_error
                    self.omegas[j] = w_proposal
                    self.frequences = new_frequencies

    def __sample_vs(self):
        actual__log_error = self.__compute_log_model_evidence(self.frequences)
        for j in range(self.M2):
            Zj = self.Z2[j]

            if self.oneDimension:
                v_proposal = sc.random.normal(self.means_v[Zj], np.sqrt(1. / self.S_v[Zj]), size=1)[0]
            else:
                v_proposal = sc.random.multivariate_normal(self.means_v[Zj], np.linalg.inv(self.S_v[Zj]), size=1)[0]

            new_frequencies = bank.getfrequences_new_vs(self, v_proposal, j)
            new_log_error = self.__compute_log_model_evidence(new_frequencies)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.vs[j] = v_proposal
                self.frequences = new_frequencies
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Acept with certain probability
                    actual__log_error = new_log_error
                    self.vs[j] = v_proposal
                    self.frequences = new_frequencies

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

    def learn_kernel(self, number_swaps):
        for i in range(number_swaps):
            self.sampling_Z()
            self.sampling_mu_sigma_omega()
            self.sampling_mu_sigma_v()
            self.__sample_omega()
            self.__sample_vs()
            # self.__sample_priors()

        self.sample_beta_sigma()
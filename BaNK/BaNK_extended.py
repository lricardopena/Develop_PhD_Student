'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import random
from scipy.stats import beta
import numpy as np
import scipy as sc
import scipy.special
from BaNK import bank


class bank_regression(bank):
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
            an = a0 + self.N/2.
            invLamba0 = alpha_prior*np.identity(self.M*2)
            invLamban = Phi_x.T.dot(Phi_x) + invLamba0
            Lamdan = np.linalg.inv(invLamban)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(invLamban).dot(miun))
        else:
            miu0 = np.zeros(self.M * 2 + 1)
            Phi_x = self.__matrix_phi(omegas)
            an = a0 + self.N / 2.
            Lamda0 = 1. / alpha_prior * np.identity(self.M * 2 + 1)
            invLamba0 = alpha_prior * np.identity(self.M * 2)
            invLambdan = Phi_x.T.dot(Phi_x) + np.linalg.inv(Lamda0)
            Lamdan = np.linalg.inv(invLambdan)
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLamba0).dot(miu0) - miun.T.dot(
                invLambdan).dot(miun))
        result_log = 1./2*np.linalg.slogdet(Lamdan)[1] + a0*np.log(b0) + sc.special.gammaln(an)
        result_log +=- an * np.log(bn) - sc.special.gammaln(a0) - (self.M * 2 + 1) / 2. * np.log(1. / alpha_prior)
        return result_log

    def __sample_omega(self):

        actual__log_error = self.__compute_log_model_evidence(self.omegas)
        for j in range(self.M):
            Zj = self.Z[j]
            if self.oneDimension:
                w_proposal = sc.random.normal(self.means[Zj], np.sqrt(1. / self.S[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.means[Zj], np.linalg.inv(self.S[Zj]), size=1)[0]
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
                if p >= u:  # Acept with certain probability
                    actual__log_error = new_log_error
                    self.omegas = w_new

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
            self.sampling_mu_sigma()
            self.__sample_omega()
            # self.__sample_priors()

        # self.sample_beta_sigma()
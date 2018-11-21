import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

'''
This implementation is made By: Msc Luis Ricardo Pena Llamas, this code is an implementation of a classification in kernel lerning.
'''

import scipy as sc
from BaNK import bank


class bank_classification(bank):

    def __compute_log_model_evidence(self, omegas, alpha_prior=0.000001, miu0 = None):
        '''
        :param omegas: the matrix with all w in W
        :param a0: loc prior for sigma value
        :param b0: scale prior for sigma value
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        '''
        Phi_x = bank.matrix_phi(self, omegas)
        if miu0 is None:
            # Lamda0 = 1./alpha_prior*np.identity(self.M*2 + 1)
            invLamba0 = alpha_prior*np.identity(self.M*2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            # miu0 = np.zeros(self.M * 2 + 1)
            invLamba0 = alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        result_log = 0
        result_mu = Phi_x.dot(miun)

        for x, y in zip(result_mu, self.Y):
            log_p_y = x - np.log(np.exp(x) + 1)
            if not (y == 1):
                log_p_y -= log_p_y
            result_log += log_p_y
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

    def learn_kernel(self, number_swaps):
        for i in range(number_swaps):
            self.sampling_Z()
            self.sampling_mu_sigma()
            self.__sample_omega()
            # self.__sample_priors()

        # self.sample_beta_sigma()

    def predict_new_X(self, X, omegas=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0 = None):
        if omegas is None:
            omegas = self.omegas

        Phi_x = bank.matrix_phi_with_X(self, omegas, self.X)

        if miu0 is None:
            invLamba0 = alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            # miu0 = np.zeros(self.M * 2 + 1)
            invLamba0 = alpha_prior * np.identity(self.M * 2)
            Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        Phi_x = bank.matrix_phi_with_X(self, omegas, X)
        return np.round(Phi_x.dot(miun),0).astype(int)
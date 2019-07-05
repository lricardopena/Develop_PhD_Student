"""
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of the proposal made by
Thesis when try to learn the Locally Stationary Kernel for my doctoral:
"""

import numpy as np
import scipy as sc
import scipy.special
from Develop_PhD_Student.BaNK.BaNK import bank
from Develop_PhD_Student.BaNK.BaNK import bank_locally_stationary
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd


class bank_regression(bank):
    def __compute_log_model_evidence(self, omegas, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        # The complexity of this is O(N d M + M^2 N + M^3)
        """
        :param omegas: the matrix with all w in W
        :param a0: loc prior for sigma value
        :param b0: scale prior for sigma value
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        """
        Phi_x = self.matrix_phi(omegas)  # complexity = O(N d M)
        an = a0 + self.N / 2.
        invLamba0 = alpha_prior * np.identity(self.M * 2)
        invLamban = Phi_x.T.dot(Phi_x) + invLamba0  # complexity = O(M^2 N)
        Lamdan = np.linalg.inv(invLamban)  # Complexity = O(M^3)
        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(invLamban).dot(miun))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLamba0).dot(miu0) - miun.T.dot(
                invLamban).dot(miun))
        result_log = 1./2*np.linalg.slogdet(Lamdan)[1] + a0*np.log(b0) + sc.special.gammaln(an)
        result_log += - an * np.log(bn) - sc.special.gammaln(a0) - (self.M * 2) / 2. * np.log(1. / alpha_prior)
        del Phi_x, Lamdan, invLamba0, invLamban, miun, omegas, a0, alpha_prior, an, b0, bn, miu0
        return result_log

    def __sample_omega(self):
        # The complexity of this if O(M(N d M + M^2 N + M^3)) = O(N d M^2 + M^3 N + M^4)
        # The complexity of this is O(N d M + M^2 N + M^3)
        actual__log_error = self.__compute_log_model_evidence(self.omegas)
        for j in range(self.M):
            Zj = self.Z[j]
            if self.oneDimension:
                w_proposal = sc.random.normal(self.means[Zj], np.sqrt(1. / self.S[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.means[Zj], np.linalg.inv(self.S[Zj]), size=1)[0]
            w_new = self.omegas.copy()
            w_new[j] = w_proposal
            # The complexity of this is O(N d M + M^2 N)
            new_log_error = self.__compute_log_model_evidence(w_new)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Accept with certain probability
                    actual__log_error = new_log_error
                    self.omegas = w_new
                del p, u
            del Zj, new_log_error, result, w_new, w_proposal

        self.sample_priors()

    @staticmethod
    def __matrix_phi_with_X(omegas, X):
        argument = X.dot(omegas.T)  # complexity = O(N d M)
        return 1. / np.sqrt(len(omegas)) * np.column_stack((np.cos(argument), np.sin(argument)))

    def __matrix_phi(self, omegas):
        argument = self.X.dot(omegas.T)  # complexity = O(N d M)
        return 1. / np.sqrt(len(omegas)) * np.column_stack((np.cos(argument), np.sin(argument)))

    def learn_kernel(self, number_swaps, X=None, y=None, name_=""):
        # The complexity of this is O(swaps(KM + K + N d M^2 + M^3 N + M^4))
        if X is not None:
            print("MSE initial of " + name_ + " " + str(self.return_mse(X, y)))
            self.save_prediction(X, y, name_ + " initial .csv")
        for i in range(number_swaps):
            self.sampling_Z()  # O(K M)
            self.sampling_mu_sigma()  # O(K)
            self.__sample_omega()  # O(N d M^2 + M^3 N + M^4)
            if X is not None:
                print(str(i) + "/" + str(number_swaps) + " MSE  of " + name_ + ": " + str(self.return_mse(X, y)))
                self.save_prediction(X, y, name_ + " swap " + str(i) + " .csv")
            # self.__sample_priors()


        # self.sample_beta_sigma()

    def return_mse(self, X, y_true):
        y_predict = self.predict(X)
        return mean_squared_error(y_true, y_predict)

    def return_r2_score(self, X, y_true):
        y_predict = self.predict(X)
        return r2_score(y_true, y_predict)

    def save_prediction_noTrue(self, X, nameFile):
        y_pred = self.predict(X)
        Y = y_pred
        X = np.column_stack((X, Y))
        df = pd.DataFrame(X)
        colum_x = X.shape[1]
        column_indices = [colum_x - 1]
        new_names = ['y_predict']
        old_names = df.columns[column_indices]
        df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        df.to_csv(nameFile)
        del df, y_pred, X, Y

    def save_prediction(self, X, y_true, nameFile):
        y_pred = self.predict(X)
        Y = np.column_stack((y_true, y_pred))
        X = np.column_stack((X, Y))
        df = pd.DataFrame(X)
        colum_x = X.shape[1]
        column_indices = [colum_x - 2, colum_x - 1]
        new_names = ['y_true', 'y_predict']
        old_names = df.columns[column_indices]
        df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        df.to_csv(nameFile)
        del df, y_pred, y_true, X, Y


class Bank_Locally_Regression(bank_locally_stationary):
    def __compute_log_model_evidence(self, frequences, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        # The complexity of this is O(N d M + M^2 N + M^3)
        """
        :param frequences: the matrix with all w in W
        :param a0: loc prior for sigma value
        :param b0: scale prior for sigma value
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        """
        Phi_x = self.matrix_phi(frequences)  # complexity = O(N d M)
        an = a0 + self.N / 2.
        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2)
        invLamban = Phi_x.T.dot(Phi_x) + invLamba0  # Complexity = O(M^2 N)
        Lamdan = np.linalg.inv(invLamban)  # Complexity = O(M^3)
        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
            bn = b0 + 1./2*(self.Y.dot(self.Y) - miun.T.dot(invLamban).dot(miun))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))
            bn = b0 + 1. / 2 * (self.Y.dot(self.Y) + miu0.T.dot(invLamba0).dot(miu0) - miun.T.dot(
                invLamban).dot(miun))
        result_log = 1./2*np.linalg.slogdet(Lamdan)[1] + a0*np.log(b0) + sc.special.gammaln(an)
        result_log += - an * np.log(bn) - sc.special.gammaln(a0) - (self.M1*self.M2 * 2) / 2. * np.log(1. / alpha_prior)
        return result_log

    def __sample_omega(self):
        # The complexity of this method if O(M(N d M + M^2 N + M^3)) = O(N d M^2 + M^3 N + M^4)

        # The complexity of this is O(N d M + M^2 N + M^3)
        actual__log_error = self.__compute_log_model_evidence(self.frequences)
        for j in range(self.M1):
            Zj = self.k_stationary.Z[j]
            if self.oneDimension:
                w_proposal = sc.random.normal(self.k_stationary.means[Zj],
                                              np.sqrt(1. / self.k_stationary.S[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.k_stationary.means[Zj],
                                                           np.linalg.inv(self.k_stationary.S[Zj]), size=1)[0]
            w_new = self.k_stationary.omegas.copy()
            w_new[j] = w_proposal
            new_frequences = self.get_frequence(w_new, self.k_positive.omegas)

            # The complexity of this is O(N d M + M^2 N)
            new_log_error = self.__compute_log_model_evidence(new_frequences)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.k_stationary.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Accept with certain probability
                    actual__log_error = new_log_error
                    self.k_stationary.omegas = w_new

        self.update_frequences()
        # The complexity of this is O(N d M + M^2 N + M^3)
        actual__log_error = self.__compute_log_model_evidence(self.frequences)
        for j in range(self.M2):
            Zj = self.k_positive.Z[j]
            if self.oneDimension:
                w_proposal = sc.random.normal(self.k_positive.means[Zj], np.sqrt(1. / self.k_positive.S[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.k_positive.means[Zj],
                                                           np.linalg.inv(self.k_positive.S[Zj]), size=1)[0]
            w_new = self.k_positive.omegas.copy()
            w_new[j] = w_proposal
            new_frequences = self.get_frequence(self.k_stationary.omegas, w_new)

            # The complexity of this is O(N d M + M^2 N)
            new_log_error = self.__compute_log_model_evidence(new_frequences)
            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.k_positive.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Accept with certain probability
                    actual__log_error = new_log_error
                    self.k_positive.omegas = w_new

        self.update_frequences()
        self.sample_priors()

    @staticmethod
    def __matrix_phi_with_X(frequences, X):
        argument = X.dot(frequences.T)  # complexity = O(N d M)
        return 1. / np.sqrt(len(frequences)) * np.column_stack((np.cos(argument), np.sin(argument)))

    def __matrix_phi(self, frequences):
        argument = self.X.dot(frequences.T)  # complexity = O(N d M)
        return 1. / np.sqrt(len(frequences)) * np.column_stack((np.cos(argument), np.sin(argument)))

    def learn_kernel(self, number_swaps, X=None, y=None, name_=""):
        # The complexity of this is O(swaps(KM + K + N d M^2 + M^3 N + M^4))

        if X is not None:
            print("Initial MSE  of " + name_ + str(self.return_mse(X, y)))
        for i in range(number_swaps):
            self.sampling_Z()  # O(K M)
            self.sampling_mu_sigma()  # O(K)
            self.__sample_omega()  # O(N d M^2 + M^3 N + M^4)
            # self.__sample_priors()
            if X is not None:
                print(str(i) + "/" + str(number_swaps) + " MSE of " + name_ + ": " + str(self.return_mse(X, y)))

        # self.sample_beta_sigma()

    def return_mse(self, X, y_true):
        y_predict = self.predict(X)
        return mean_squared_error(y_true, y_predict)

    def return_r2_score(self, X, y_true):
        y_predict = self.predict(X)
        return r2_score(y_true, y_predict)

    def save_prediction(self, X, y_true, nameFile):
        y_pred = self.predict(X)
        Y = np.column_stack((y_true, y_pred))
        X = np.column_stack((X, Y))
        df = pd.DataFrame(X)
        colum_x = X.shape[1]
        column_indices = [colum_x - 3, colum_x - 2, colum_x - 1]
        new_names = ['y_true', 'y_predict']
        old_names = df.columns[column_indices]
        df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        df.to_csv(nameFile)

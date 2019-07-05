"""
This implementation is made By: Msc Luis Ricardo Pena Llamas, this code is an implementation of a classification in kernel lerning.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import gc
import scipy as sc
from Develop_PhD_Student.BaNK.BaNK import bank, bank_locally_stationary
from scipy.special import expit
import pandas as pd
import time
import matplotlib.pyplot as plt


class bank_classification(bank):
    def __compute_log_model_evidence(self, omegas, alpha_prior=0.000001, miu0=None):
        """
        :param omegas: the matrix with all w in W
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        """

        Phi_x = self.matrix_phi(omegas)
        invLamba0 = alpha_prior * np.identity(self.M * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0) # complexity = O(M^2 N + M^3)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        result_mu = self.sigmoid(Phi_x.dot(miun))

        # new log likelihood
        result_log_new = - np.sum(self.Y * np.log(result_mu) + (1 - self.Y) * np.log(1 - result_mu))
        del Lamdan, Phi_x, alpha_prior, invLamba0, miu0, miun, omegas, result_mu
        return result_log_new

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
                del u, p

            del Zj, new_log_error, result, w_new, w_proposal
        del actual__log_error

    def learn_kernel(self, number_swaps, allX=None, allY=None, name=""):
        print_score = not(allX is None)
        if print_score:
            print("Initial:   " + name + " full X score: " + str(
                self.score(allX, allY)) + "   Time: " + str(time.strftime("%c")))

        for i in range(number_swaps):
            self.sampling_Z()
            self.sampling_mu_sigma()
            self.__sample_omega()
            self.sample_priors()
            gc.collect()
            if print_score:
                print("Swap: " + str(i) + "/" + str(number_swaps) + "  " + name + " full X score: " + str(
                    self.score(allX, allY)) + "   Time: " + str(time.strftime("%c")))
                print("Confussion matrix full set: ")
                print(self.return_confussion_matrix(allX, allY))

        # self.sample_beta_sigma()

    def predict_X(self, X, omegas=None):
        if omegas is None:
            omegas = self.omegas

        self.sample_beta_sigma()
        Y = []
        Phi_X = self.get_Phi_X(X, omegas)
        for x in Phi_X.dot(self.beta):
            Y.append(1. / (1 - np.exp(-x)))

        Y = np.round(Y, 0).astype(int)
        Y[np.where(Y <= 0)[0]] = 0
        Y[np.where(Y > 1)[0]] = 1
        del Phi_X, omegas
        return Y

    def predict_new_X(self, X, omegas=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        """

        :type X: np.array
        """
        if omegas is None:
            omegas = self.omegas

        Phi_x = self.matrix_phi_with_X(omegas, self.X)
        invLamba0 = alpha_prior * np.identity(self.M * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        Phi_x = self.matrix_phi_with_X(omegas, X)
        # return np.round(((1/(1+np.exp(-Phi_x.dot(miun)))) - .122),0)
        # return np.round(((1 / (1 + np.exp(-Phi_x.dot(miun))))), 0).astype(int)
        # return np.round(result/(1 - result), 0).astype(int)
        # result = Phi_x.dot(miun)
        # return np.round(1 / (1 + np.exp(-result)), 0).astype(int)
        result = np.round(Phi_x.dot(miun).astype(np.double), 0).astype(int)
        result[np.where(result < 0)] = 0
        result[np.where(result > 1)] = 1
        del Phi_x, miun, invLamba0, Lamdan, omegas, X, alpha_prior, miu0
        gc.collect()
        return result

    def __predict_new_X(self, X, omegas=None, alpha_prior=0.000001, miu0=None):
        if omegas is None:
            omegas = self.omegas

        Phi_x = bank.matrix_phi_with_X(self, omegas, self.X)
        invLamba0 = alpha_prior * np.identity(self.M * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        Phi_x = bank.matrix_phi_with_X(self, omegas, X)
        return Phi_x.dot(miun)

    def predict_proba_X(self, X, omegas=None):
        if omegas is None:
            omegas = self.omegas

        self.sample_beta_sigma()
        Y = []
        Phi_X = self.get_Phi_X(X, omegas)
        for x in Phi_X.dot(self.beta):
            Y.append(1. / (1 - np.exp(-x)))

        return Y

    def predict_proba(self, X):
        alpha_prior = 0.000001
        Phi_x = self.matrix_phi_with_X(self.omegas, self.X)
        invLamba0 = alpha_prior * np.identity(self.M * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
        miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        Phi_x = self.matrix_phi_with_X(self.omegas, X)
        result = Phi_x.dot(miun)
        result = 1 / (1 + np.exp(-result))
        return result

    def score(self, X, y_true):
        y_pred = self.predict_new_X(X)
        # y_pred = self.predict_X(X)
        res = accuracy_score(y_pred, y_true.astype(int))
        del X, y_pred, y_true
        return res

    def return_confussion_matrix(self, X, y_true):
        y_pred = self.predict_new_X(X)
        # y_pred = self.predict_X(X)
        # y_predict_proba = np.array(self.predict_proba(X), dtype=int)
        #return confusion_matrix(y_true, y_predict_proba)
        conf_matrix = confusion_matrix(y_true, y_pred)
        del y_pred, X, y_true
        return conf_matrix

    def save_prediction(self, X, y_true, nameFile):
        y_pred = self.predict_new_X(X)
        y_proba =self.predict_proba(X)
        # y_pred = self.predict_X(X)
        # y_proba = self.predict_proba_X(X)
        Y = np.column_stack((y_true, y_pred))
        Y = np.column_stack((Y, y_proba))
        X = np.column_stack((X, Y))
        df = pd.DataFrame(X)
        colum_x = X.shape[1]
        column_indices = [colum_x - 3, colum_x - 2, colum_x - 1]
        new_names = ['y_true', 'y_predict', 'y_proba']
        old_names = df.columns[column_indices]
        df.rename(columns=dict(zip(old_names, new_names)), inplace = True)
        df.to_csv(nameFile)

    def return_ROC_curve(self, X_test, y_test, plot=False):
        y_predict_proba = self.predict_proba(X_test)
        # y_predict_proba = self.predict_proba_X(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
        print("AUC: " + str(auc(fpr, tpr)))
        if plot:
            plt.plot(fpr, tpr, label='RT + LR')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()

    @staticmethod
    def sigmoid(x):
        return expit(x)


class bank_locally_classification(bank_locally_stationary):
    def __compute_log_model_evidence(self, omegas, vs, alpha_prior=0.000001, miu0=None):
        """
        :param frequences: the matrix with all w in W
        :param alpha_prior: scale prior for beta
        :param miu0: mean prior for beta
        :return: the error with omegas corresponding
        """
        Phi_x = self.matrix_phi(self.get_frequence(omegas, vs))
        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)  # complexity = O(M^2 N + M^3)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        Phi_x = self.matrix_phi_with_X(self.get_frequence_minus(omegas, vs), self.X )
        result_mu = Phi_x.dot(miun)
        result_log_new = - np.sum(self.Y * np.log(result_mu) + (1 - self.Y) * np.log(1 - result_mu))
        del Lamdan, Phi_x, alpha_prior, omegas, vs, invLamba0, miun, miu0, result_mu
        return result_log_new

    def __sample_omega(self):
        actual__log_error = self.__compute_log_model_evidence(self.get_omegas(), self.get_vs())
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
            new_log_error = self.__compute_log_model_evidence(w_new, self.get_vs())

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.k_stationary.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Acept with certain probability
                    actual__log_error = new_log_error
                    self.k_stationary.omegas = w_new
                del u, p
            del new_log_error, Zj, result, w_proposal, w_new
        del actual__log_error
        # self.update_frequences()
        actual__log_error = self.__compute_log_model_evidence(self.get_omegas(), self.get_vs())
        for j in range(self.M2):
            Zj = self.k_positive.Z[j]
            if self.oneDimension:
                w_proposal = sc.random.normal(self.k_positive.means[Zj], np.sqrt(1. / self.k_positive.S[Zj]), size=1)[0]
            else:
                w_proposal = sc.random.multivariate_normal(self.k_positive.means[Zj],
                                                           np.linalg.inv(self.k_positive.S[Zj]), size=1)[0]
            w_new = self.k_positive.omegas.copy()
            w_new[j] = w_proposal
            new_log_error = self.__compute_log_model_evidence(self.get_omegas(), w_new)

            result = new_log_error - actual__log_error
            if result > 0:
                actual__log_error = new_log_error
                self.k_positive.omegas = w_new
            else:
                u = np.random.rand(1)[0]
                p = np.exp(result)
                if p >= u:  # Acept with certain probability
                    actual__log_error = new_log_error
                    self.k_positive.omegas = w_new
                del u, p
            del new_frequences, new_log_error, Zj, result, w_proposal, w_new
        del actual__log_error
        self.update_frequences()
        # self.sample_priors()

    def learn_kernel(self, number_swaps, allX = None, allY = None, name=""):
        print_score = not (allX is None)
        if print_score:
            print("Initial:   " + name + " full X score: " + str(
                self.score(allX, allY)) + "   Time: " + str(time.strftime("%c")))
        for i in range(number_swaps):
            self.sampling_Z()
            self.sampling_mu_sigma()
            self.__sample_omega()
            # self.sample_priors()
            gc.collect()
            if print_score:
                print("Swap: " + str(i) + "/" + str(number_swaps) + "  " + name + " full X score: " + str(
                    self.score(allX, allY)) + "   Time: " + str(time.strftime("%c")))
                print("Confussion matrix full set: ")
                print(self.return_confussion_matrix(allX, allY))

        # self.sample_beta_sigma()

    def predict_X(self, X, frequences=None):
        if frequences is None:
            frequences = self.frequences

        self.sample_beta_sigma()
        Y = []
        Phi_X = self.get_Phi_X(X, frequences)
        for x in Phi_X.dot(self.beta):
            Y.append(1. / (1 - np.exp(-x)))

        return np.round(Y, 0).astype(int)

    def predict_new_X(self, X, frequences=None, a0=0.001, b0=0.001, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequences
        Phi_x = self.matrix_phi(frequences)
        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        del Phi_x

        Phi_x = self.matrix_phi_with_X(self.get_frequence_minus(), X)
        result = np.round(Phi_x.dot(miun), 0).astype(int)
        index = np.where(result < 0)
        result[index] = 0
        result[np.where(result > 1)] = 1
        del Phi_x, frequences, miu0, miun, index, invLamba0, Lamdan, X, alpha_prior
        return result

    def predict_new_X_proba(self, X, frequences=None, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequences

        Phi_x = self.matrix_phi_with_X(frequences, self.X)
        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        del Phi_x
        Phi_x = self.matrix_phi_with_X(frequences, X)
        # return np.round(((1/(1+np.exp(-Phi_x.dot(miun)))) - .122),0)
        # return np.round(((1 / (1 + np.exp(-Phi_x.dot(miun))))), 0).astype(int)
        # return np.round(result/(1 - result), 0).astype(int)
        # result = Phi_x.dot(miun)
        # return np.round(1 / (1 + np.exp(-result)), 0).astype(int)
        res = Phi_x.dot(miun)
        del Phi_x, invLamba0, Lamdan, frequences, miun, miu0
        return res

    def __predict_new_X(self, X, frequences=None, alpha_prior=0.000001, miu0=None):
        if frequences is None:
            frequences = self.frequences

        Phi_x = self.matrix_phi_with_X(frequences, self.X)

        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)

        if miu0 is None:
            miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        else:
            miun = Lamdan.dot(invLamba0.dot(miu0) + Phi_x.T.dot(self.Y))

        del Phi_x
        Phi_x = self.matrix_phi_with_X(frequences, X)
        del frequences
        res = Phi_x.dot(miun)
        del Phi_x, X, alpha_prior, miu0, miun, Lamdan, invLamba0
        return res

    def predict_proba_X(self, X, frequences=None):
        if frequences is None:
            frequences = self.frequences

        self.sample_beta_sigma()
        Y = []
        Phi_X = self.get_Phi_X(X, frequences)
        for x in Phi_X.dot(self.beta):
            Y.append(1. / (1 - np.exp(-x)))
        del frequences, Phi_X
        return Y

    def predict_proba(self, X):
        alpha_prior = 0.000001
        Phi_x = self.matrix_phi_with_X(self.frequences, self.X)
        invLamba0 = alpha_prior * np.identity(self.M1 * self.M2 * 2 + self.bias)
        Lamdan = np.linalg.inv(Phi_x.T.dot(Phi_x) + invLamba0)
        miun = Lamdan.dot(Phi_x.T.dot(self.Y))
        Phi_x = self.matrix_phi_with_X(self.frequences, X)
        result = Phi_x.dot(miun)
        result = 1 / (1 + np.exp(-result))
        del alpha_prior, Phi_x, invLamba0, Lamdan, miun
        return result

    def score(self, X, y_true):
        y_pred = self.predict_new_X(X)
        res = accuracy_score(y_true, y_pred)
        del X, y_pred, y_true
        return res

    def return_confussion_matrix(self, X, y_true):
        y_pred = self.predict_new_X(X)
        res = confusion_matrix(y_true, y_pred)
        del y_pred, y_true, X
        return res

    def save_prediction(self, X, y_true, nameFile):
        y_pred = self.predict_new_X(X)
        y_proba = self.predict_proba(X)
        Y = np.column_stack((y_true, y_pred))
        Y = np.column_stack((Y, y_proba))
        X = np.column_stack((X, Y))
        df = pd.DataFrame(X)
        colum_x = X.shape[1]
        column_indices = [colum_x - 3, colum_x - 2, colum_x - 1]
        new_names = ['y_true', 'y_predict', 'y_proba']
        old_names = df.columns[column_indices]
        df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        df.to_csv(nameFile)
        del X, y_true, y_pred, y_proba, Y, df, colum_x, column_indices, new_names, old_names

    def return_ROC_curve(self, X_test, y_test, plot=False):
        y_predict_proba = self.predict_proba(X_test)
        del X_test
        fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
        del y_test, y_predict_proba
        if plot:
            plt.plot(fpr, tpr, label='RT + LR')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()

        return fpr, tpr

    @staticmethod
    def sigmoid(x):
        return expit(x)

import datetime
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from matplotlib import patches
from numba import jit
from scipy.stats import beta
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from BaNK.data_test.classification import Classification


def samplingGMM_1d(N, means, cov, pi):
    sampling = np.zeros(N)
    U = np.random.uniform(0, 1, size=N)
    for i in range(N):
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > U[i]:
                xi = np.random.normal(means[index], cov[index])
                sampling[i] = xi
                break
            index += 1

    return sampling


@jit
def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, means.shape[1]), dtype=float)
    U = np.random.uniform(0, 1, size=N)
    for i in range(N):
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > U[i]:
                sampling[i] = sc.random.multivariate_normal(means[index], cov[index], size=1)[0]
                break
            index += 1

    return sampling


def phi_xi(x, w):
    argument = w.dot(x).T

    return 1. / np.sqrt(len(w)) * np.concatenate((np.cos(argument), np.sin(argument)))


def __matrix_phi(X, omegas, oneDimension):
    if oneDimension:
        argument = X.reshape(len(X), 1).dot(omegas.reshape(1, len(omegas)))
    else:
        argument = X.dot(omegas.T)
    return 1. / np.sqrt(len(omegas)) * np.column_stack((np.cos(argument), np.sin(argument)))


def f(X, omegas, beta, oneDimension):
    Phi_x = __matrix_phi(X, omegas, oneDimension)

    means = np.array(Phi_x).dot(beta)
    Y = []
    for u in means:
        Y.append(np.random.normal(u, 1, size=1)[0])
    # Y = np.random.multivariate_normal(mean=means_omega, cov=np.identity(len(means_omega)), size=1)[0]
    return np.array(Y)


def printKernel(X, means, sigmas, pik):
    Y = np.zeros_like(X)
    for i in range(len(means)):
        Y += pik[i] * np.exp(-1. / 2 * (sigmas[i] * X ** 2)) * np.cos(means[i] * X)
    return Y


def example_1D_regression():
    means = np.array([0, 3 * np.pi / 4, 11 * np.pi / 8])
    cov = np.array([1. / 4, 1. / 4, 1. / 4 ** 2])
    realpik = np.array([1. / 3, 1. / 3, 1. / 3])

    # means = np.array([0, 3./4 * np.pi ])
    # cov = np.array([1. / 2**2, 1. / 2**2])
    # realpik = np.array([1. / 2, 1. / 2])
    N = 8000
    M = 250
    inicio, fin = -100, 100

    real_omegas = samplingGMM_1d(N=M, means=means, cov=np.sqrt(cov), pi=realpik)
    real_beta = np.array(multivariate_normal.rvs(mean=np.zeros(2 * M), cov=np.identity(2 * M), size=1))
    # real_beta = np.append(1, real_beta)
    Xi = np.linspace(inicio, fin, N)
    Xi = Xi.reshape(N, 1)
    Yi = f(Xi, real_omegas, real_beta, True)
    X_train, X_test, Y_train, Y_test = train_test_split(Xi, Yi, test_size=0.2)
    bank = BaNK_regression.bank_regression(X_train, Y_train, M)
    real_Phi_X = bank.matrix_phi_with_X(real_omegas, Xi)
    plt.scatter(Xi, real_Phi_X.dot(real_beta), label='Real mean', color='black')
    # plt.scatter(Xi, real_Phi_X.dot(real_beta) + 3 * 1, color='black', label='Real variance')
    # plt.scatter(Xi, real_Phi_X.dot(real_beta) - 3 * 1, color='black')
    plt.legend()
    plt.show()

    number_of_rounds = 5

    bank.learn_kernel(number_of_rounds)
    Xi = np.linspace(-25, 25, N)

    Yi_learned = printKernel(Xi, bank.means, bank.get_covariance_matrix(), bank.get_pik())

    Yi_real = printKernel(Xi, means, cov, realpik)

    plt.plot(Xi, Yi_learned, label='Kernel learned')
    plt.plot(Xi, Yi_real, label='Kernel real')
    plt.legend()
    plt.show()
    Xi = Xi.reshape(len(Xi), 1)
    real_Phi_X = bank.matrix_phi_with_X(real_omegas, Xi)
    Phi_X = bank.matrix_phi_with_X(bank.omegas, Xi)
    error = np.abs(real_Phi_X - Phi_X)
    print("Error Phi_X: " + str(error.mean()) + " +- " + str(error.std()))
    Yi_pred = bank.predict_new_X_beta_omega(X_train)
    error = np.abs(Y_train - Yi_pred)
    print("Error Yi_train: " + str(error.mean()) + " +- " + str(error.std()))
    # error = Phi_X.dot(bank.beta) - real_Phi_X.dot(real_beta)
    # print("MSE 1D: " + str(np.abs(error).mean()))
    # plt.scatter(Xi, Yi, label='Real Yi')
    # plt.scatter(Xi, Yi_predicted, label='Predicted Yi')
    Yi_pred = bank.predict_new_X_beta_omega(X_test)
    bank.sample_beta_sigma()
    sigma_e = bank.sigma_e
    error = np.abs(Y_test - Yi_pred)
    print("Error Yi_test: " + str(error.mean()) + " +- " + str(error.std()))
    Xi = np.linspace(inicio, fin, N)
    Xi = Xi.reshape(N, 1)
    # Phi_X.dot(self.beta)
    real_Phi_X = bank.matrix_phi_with_X(real_omegas, Xi)
    Phi_X = bank.matrix_phi_with_X(bank.omegas, Xi)
    plt.plot(Xi, real_Phi_X.dot(real_beta), label='Real mean', color='black')
    plt.plot(Xi, real_Phi_X.dot(real_beta) + 3 * 1, '--', color='black', label='Real variance')
    plt.plot(Xi, real_Phi_X.dot(real_beta) - 3 * 1, '--', color='black')
    plt.plot(Xi, Phi_X.dot(bank.beta), label='Sample mean', color='red')
    plt.plot(Xi, Phi_X.dot(bank.beta) + 3 * np.sqrt(sigma_e), '--', color='red', label='Sampled variance')
    plt.plot(Xi, Phi_X.dot(bank.beta) - 3 * np.sqrt(sigma_e), '--', color='red')
    plt.legend()
    plt.show()


def example_2d_regression():
    means = np.array([[-1, 0], [3. * np.pi / 4, 11. * np.pi / 8]])
    cov = np.array([[[1 / 2., 0], [0, 1 / 3.]], [[1. / 4, 1. / -5], [1. / -5, 1. / 5.3]]])
    realpik = np.array([1. / 2, 1. / 2])
    N = 1000
    M = 250
    real_omegas = samplingGMM(N=M, means=means, cov=cov, pi=realpik)
    # plt.scatter(real_omegas.T[0], real_omegas.T[1], label='Samples')
    # plt.legend()
    # plt.show()
    real_beta = np.array(multivariate_normal.rvs(mean=np.zeros(2 * M), cov=np.identity(2 * M), size=1))
    # Xi = np.linspace(-10, 10, 2000)
    X = np.linspace(-10, 10, N)
    Y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(X, Y)
    Xi = sc.random.multivariate_normal([0, 0], [[10, 0], [0, 10]], N)
    Yi = f(Xi, real_omegas, real_beta)
    # plt.scatter(Xi, Yi, label='Samples')
    # plt.legend()
    # plt.show()
    number_of_rounds = 10
    bank = BaNK_regression.bank_regression(Xi, Yi, M)
    bank.learn_kernel(number_of_rounds)
    bank.sample_beta_sigma()
    # Yi = f(Xi, real_omegas, real_beta)
    real_Phi_X = __matrix_phi(Xi, real_omegas)
    Phi_X = __matrix_phi(Xi, bank.omegas)
    # Yi_predicted = np.random.multivariate_normal(Phi_X.dot(bank.beta), sigma_e * np.identity(len(Phi_X)))
    error = Phi_X.dot(bank.beta) - real_Phi_X.dot(real_beta)
    print("MSE 2D: " + str(np.abs(error).mean()))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(Xi.T[0], Xi.T[1], real_Phi_X.dot(real_beta), linewidth=0.2, antialiased=True, label='Real',
               color='blue')
    ax.scatter(Xi.T[0], Xi.T[1], Phi_X.dot(bank.beta), linewidth=0.2, antialiased=True, label='Sampled',
               color='red')
    # ax.legend()
    # plt.legend()
    plt.show()


def print_ggm_1D(real_parameters, computed_parameters, separatedGaussians=True):
    real_mean, real_cov, real_pik = real_parameters[0], np.sqrt(real_parameters[1]), real_parameters[2]
    mean_sampled, cov_sampled, pik_sampled = computed_parameters[0], np.sqrt(computed_parameters[1]), \
                                             computed_parameters[2]

    # real parameters
    min_mean = min(real_mean)
    max_mean = max(real_mean)
    max_variance = max(real_cov)

    starX = min_mean - 4 * max_variance
    stopX = max_mean + 4 * max_variance
    granularity = 10000
    X = np.linspace(starX, stopX, granularity)
    Y = np.zeros_like(X)
    for i in range(len(real_pik)):
        mean, sigma, pik = real_mean[i], real_cov[i], real_pik[i]

        if separatedGaussians:
            X = np.linspace(mean - 4 * sigma, mean + 4 * sigma, granularity)
            Y = pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)
            plt.plot(X, Y, label='Real Gaussian Mixture', color='b', linestyle='--')
        else:
            Y += pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)

    if not separatedGaussians:
        plt.plot(X, Y, label='Real Gaussian Mixture')

    # sample parameters
    min_mean = min(mean_sampled)
    max_mean = max(mean_sampled)
    max_variance = max(cov_sampled)

    starX = min_mean - 4 * max_variance
    stopX = max_mean + 4 * max_variance
    granularity = 10000
    X = np.linspace(starX, stopX, granularity)
    Y = np.zeros_like(X)
    for i in range(len(pik_sampled)):
        mean, sigma, pik = mean_sampled[i], cov_sampled[i], pik_sampled[i]

        if separatedGaussians:
            X = np.linspace(mean - 4 * sigma, mean + 4 * sigma, granularity)
            Y = pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)
            plt.plot(X, Y, label='Sampled Gaussian Mixture', color='k', linestyle='--')
        else:
            Y += pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)

    if not separatedGaussians:
        plt.plot(X, Y, label='Sampled Gaussian Mixture')
    plt.legend()

    plt.show()


def exampleGMM1D():
    means, variance, pik = np.array([-4, -50, 10]), np.array([4, 4, 4]), np.array([1. / 3, 1. / 3, 1. / 3])
    X = samplingGMM_1d(500, means, np.sqrt(variance), pik)
    inf_learn = infinite_GMM.infinite_GMM(X, initial_number_class=3)
    inf_learn.learn_GMM(600)
    print_ggm_1D(np.array([means, variance, pik]),
                 np.array([inf_learn.means, 1. / inf_learn.S, inf_learn.get_weights()]))
    print_ggm_1D(np.array([means, variance, pik]),
                 np.array([inf_learn.means, 1. / inf_learn.S, inf_learn.get_weights()]), False)
    pltgmm = printGMM.plotgaussianmixture(X, inf_learn.means, np.sqrt(1. / inf_learn.S), inf_learn.get_weights())
    pltgmm_real = printGMM.plotgaussianmixture(X, means, np.sqrt(variance), pik)
    pltgmm.print_seperategaussians()
    pltgmm_real.print_seperategaussians()
    pltgmm.printGMM()
    pltgmm_real.printGMM()
    print("some")


def exampleGMM2D():
    means, variance, pik = np.array([[0, 1], [20, 60], [100, 100], [10, 10]]), np.array(
        [[[4, 0], [0, 2]], [[4, 0], [0, 2]], [[4, 0], [0, 2]], [[4, 0], [0, 2]]]), np.array(
        [1. / 4, 1. / 4, 1. / 4, 1. / 4])
    X = samplingGMM(500, means, variance, pik)
    inf_learn = infinite_GMM.infinite_GMM(X, initial_number_class=3)
    inf_learn.learn_GMM(600)
    pltgmm = printGMM.plotgaussianmixture(X, inf_learn.means, inf_learn.get_covariance_matrix(),
                                          inf_learn.get_weights())
    pltgmm.print_seperategaussians()
    pltgmm.printGMM()
    print("some")


def example_classification_xor():
    size1 = 10000
    means = np.array([[0, 0], [10, 10]])
    covarance = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1, means, covarance, [1. / 2, 1. / 2])

    means = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1, means, covarance, [1. / 2, 1. / 2])
    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))

    plt.scatter(X1.T[0], X1.T[1], label='Class 1')
    plt.scatter(X2.T[0], X2.T[1], label='Class 2')
    plt.legend()
    plt.show()

    bank_cl = BaNK_classification.bank_classification(X, Y, 250)
    bank_cl.learn_kernel(5)

    means = np.array([[0, 0], [10, 10]])
    covarance = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1 * 10, means, covarance, [1. / 2, 1. / 2])

    means = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1 * 10, means, covarance, [1. / 2, 1. / 2])

    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))
    X = np.column_stack((X, Y))

    np.random.shuffle(X)

    X = np.random.uniform(low=-10, high=20, size=(10000, 2))

    Y_predict = bank_cl.predict_new_X(X)

    z = np.where(Y_predict == 1)
    plt.scatter(X[z].T[0], X[z].T[1], label='Class 1')

    z = np.where(Y_predict == 0)
    plt.scatter(X[z].T[0], X[z].T[1], label='Class 2')
    plt.legend()
    plt.show()


def example_classification_xor_locally():
    size1 = 1000
    means = np.array([[0, 0], [10, 10]])
    covarance = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1, means, covarance, [1. / 2, 1. / 2])

    means = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1, means, covarance, [1. / 2, 1. / 2])
    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))

    plt.scatter(X1.T[0], X1.T[1], label='Class 1')
    plt.scatter(X2.T[0], X2.T[1], label='Class 2')
    plt.legend()
    plt.show()

    bank_cl = BaNK_classification.bank_locally_classification(X, Y, 30, 35)
    bank_cl.learn_kernel(10)

    means1 = np.array([[0, 0], [10, 10]])
    covarance1 = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1 * 10, means1, covarance1, [1. / 2, 1. / 2])

    means2 = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1 * 10, means2, covarance1, [1. / 2, 1. / 2])

    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))
    X = np.column_stack((X, Y))

    np.random.shuffle(X)

    X = np.random.uniform(low=-10, high=20, size=(1000, 2))

    Y_predict = bank_cl.predict_new_X(X)

    z = np.where(Y_predict == 1)

    fig, ax = plt.subplots()
    for mean, sigma in zip(means1, covarance):
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        index_max_eigenvalue = np.where(eigenvalues == max(eigenvalues))[0][0]
        index_min_eigenvalue = 1 - index_max_eigenvalue
        angle = np.arctan(eigenvectors[index_max_eigenvalue][1] / eigenvectors[index_max_eigenvalue][0])

        if angle < 0:
            angle += 2 * np.pi

        e = patches.Ellipse(mean, 6 * np.sqrt(eigenvalues[index_max_eigenvalue]),
                            6 * np.sqrt(eigenvalues[index_min_eigenvalue]),
                            angle=angle, linewidth=2, fill=False, zorder=2)
        ax.add_patch(e)

    for mean, sigma in zip(means2, covarance):
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        index_max_eigenvalue = np.where(eigenvalues == max(eigenvalues))[0][0]
        index_min_eigenvalue = 1 - index_max_eigenvalue
        angle = np.arctan(eigenvectors[index_max_eigenvalue][1] / eigenvectors[index_max_eigenvalue][0])

        if angle < 0:
            angle += 2 * np.pi

        e = patches.Ellipse(mean, 6 * np.sqrt(eigenvalues[index_max_eigenvalue]),
                            6 * np.sqrt(eigenvalues[index_min_eigenvalue]),
                            angle=angle, linewidth=2, fill=False, zorder=2)
        ax.add_patch(e)
    fig.tight_layout()

    plt.scatter(X[z].T[0], X[z].T[1], label='Class 1')

    z = np.where(Y_predict == 0)
    plt.scatter(X[z].T[0], X[z].T[1], label='Class 2')
    plt.legend()
    plt.show()


def example_2_spiral():
    # size = 1000
    # t = np.linspace(-50, -1, size)
    #
    # a = 0.501
    # x, y = getSpiral(100, 200, t, a)
    #
    # plt.plot(x, y, label='-25 to 0')
    #
    # t = np.linspace(1, 50, size)
    #
    # a = 0.5
    # x, y = getSpiral(100.3, 200, t, a)
    # plt.plot(x, y, label='0 to 25')
    #
    # plt.legend()
    # plt.show()

    f = np.loadtxt("spiral.data")
    x, y = f[:, :2], f[:, 2]

    plt.plot(x[np.where(y == 1)].T[0], x[np.where(y == 1)].T[1])
    plt.plot(x[np.where(y == -1)].T[0], x[np.where(y == -1)].T[1])
    plt.show()
    y[np.where(y == -1)] = 0

    bank_cl = BaNK_classification.bank_classification(x, y, 250)
    bank_cl.learn_kernel(5000)

    Y_predict = bank_cl.predict_new_X(x)

    error = np.abs(y - Y_predict)

    print("Error: " + str(np.sum(error) / len(y)))

    plt.scatter(x[np.where(Y_predict == 1)].T[0], x[np.where(Y_predict == 1)].T[1])
    plt.scatter(x[np.where(Y_predict == 0)].T[0], x[np.where(Y_predict == 0)].T[1])
    plt.show()


def getSpiral(xcenter, ycenter, linespace, a):
    theta = linespace
    x = a * theta * np.cos(theta) + xcenter
    y = a * theta * np.sin(theta) + ycenter

    return x, y


def cross_validation_classification(X, Y, M, number_rounds, times_crossvalidation):
    scores = []
    n = len(X)
    n_train = int(n * 0.8)

    for i in range(times_crossvalidation):
        time_begin = datetime.datetime.now()
        print("Begin " + str(i + 1) + " of " + str(times_crossvalidation) + " at " + str(
            time_begin.strftime("%A, %d. %B %Y %I:%M%p")))

        np.random.shuffle(X)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        bank = BaNK_classification.bank_classification(X_train, Y_train, M)
        gc.collect()
        bank.learn_kernel(number_rounds)
        time_end = datetime.datetime.now()
        print("End " + str(i + 1) + " of " + str(times_crossvalidation) + " at " + str(
            time_end.strftime("%A, %d. %B %Y %I:%M%p")))
        print("Training time: " + str(time_end - time_begin))
        Y_predicted = bank.predict_new_X_beta_omega(X_test)
        error = (Y_test - Y_predicted)
        scores.append([np.sum(np.abs(error)), n])
        gc.collect()
    return np.array(scores)


def make_classification():
    # Airfoil Self-Noise Data Set
    M = 500
    df = pd.read_csv('datasets classification/Diabetic Retinopathy Debrecen.csv')
    X1 = np.array(df.T[:19].T)
    Y1 = np.array(df['Output'])

    number_of_rounds = 15
    scores_airfoil = cross_validation_classification(X1, Y1, M, number_of_rounds, 5)
    print("Scores of Retinopathy with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    for row in scores_airfoil:
        print("Missclassified retinopathy: " + str(row[0]) + " in " + str(row[1]))
    print("----------------------------------------")

    df = pd.read_csv('dataset regression/egg eye state.csv')
    X2 = np.array(df.T[:13].T)
    Y2 = np.array(df['Output'])

    number_of_rounds = 15
    scores_bike = cross_validation_classification(X2, Y2, M, number_of_rounds, 5)
    print("Scores of egg eye with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    for row in scores_bike:
        print("Missclassified egg eye: " + str(row[0]) + " in " + str(row[1]))
    print("----------------------------------------")

    df = pd.read_csv('dataset regression/pima.csv')
    X3 = np.array(df.T[:8].T)
    Y3 = np.array(df['Output'])

    number_of_rounds = 15
    scores_airfoil = cross_validation_classification(X3, Y3, M, number_of_rounds, 5)
    print("Scores of pima with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    for row in scores_airfoil:
        print("Missclassified pima: " + str(row[0]) + " in " + str(row[1]))
    print("----------------------------------------")

    df = pd.read_csv('dataset regression/skin segmentation.csv', sep='\t')
    X4 = np.array(df.T[:3].T)
    Y4 = np.array(df[3])

    number_of_rounds = 15
    scores_airfoil = cross_validation_classification(X4, Y4, M, number_of_rounds, 5)
    print("Scores of skin segmentation with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    for row in scores_airfoil:
        print("Missclassified skin segmentation: " + str(row[0]) + " in " + str(row[1]))
    print("----------------------------------------")

    print("Finish classification")


def cross_validation_regression(X, Y, M, number_rounds, times_crossvalidation):
    scores = []
    n = len(X)
    n_train = int(n * 0.8)

    for i in range(times_crossvalidation):
        time_begin = datetime.datetime.now()
        print("Begin " + str(i + 1) + " of " + str(times_crossvalidation) + " at " + str(
            time_begin.strftime("%A, %d. %B %Y %I:%M%p")))

        np.random.shuffle(X)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        bank = BaNK_regression.bank_regression(X_train, Y_train, M)
        bank.learn_kernel(number_rounds)
        time_end = datetime.datetime.now()
        print("End " + str(i + 1) + " of " + str(times_crossvalidation) + " at " + str(
            time_end.strftime("%A, %d. %B %Y %I:%M%p")))
        print("Training time: " + str(time_end - time_begin))
        Y_predicted = bank.predict_new_X_beta_omega(X_test)
        # error = (Y_test - Y_predicted) / rangey
        error = (Y_test - Y_predicted)
        error_mean = np.abs(error).mean()
        error_std = np.abs(error).std()
        scores.append([error_mean, error_std])
    return np.array(scores)


def make_regression():
    # Airfoil Self-Noise Data Set
    M = 500
    df = pd.read_csv('dataset regression/Airfoil Self-Noise Data Set.csv', sep='\t')
    X1 = np.array(df.T[:5].T)
    Y1 = np.array(df['output (Db)'])

    number_of_rounds = 45
    scores_airfoil = cross_validation_regression(X1, Y1, M, number_of_rounds, 5)
    print("Scores of Airfoil with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    print("Range: " + str(max(Y1) - min(Y1)))
    for row in scores_airfoil:
        print("MSE Airfoil: " + str(row[0]) + " +- " + str(row[1]))
    print("----------------------------------------")
    # bike dataset
    # M = 750
    df = pd.read_csv('dataset regression/bike.csv')
    X2 = np.array(df.T[:18].T)
    Y2 = np.array(df['cnt'])

    number_of_rounds = 45
    scores_bike = cross_validation_regression(X2, Y2, M, number_of_rounds, 5)
    print("Scores of bike with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    print("Range: " + str(max(Y2) - min(Y2)))
    for row in scores_bike:
        print("MSE bike: " + str(row[0]) + " +- " + str(row[1]))
    print("----------------------------------------")
    # M = 750
    df = pd.read_csv('dataset regression/Concrete_Data.csv')
    X3 = np.array(df.T[:7].T)
    Y3 = np.array(df['Concrete'])

    number_of_rounds = 45
    scores_bike = cross_validation_regression(X3, Y3, M, number_of_rounds, 5)
    print("Scores of concrete with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    print("Range: " + str(max(Y3) - min(Y3)))
    for row in scores_bike:
        print("MSE concrete: " + str(row[0]) + " +- " + str(row[1]))
    print("----------------------------------------")
    print("Finish regression")


def check_models():
    # regression
    # example_1D_regression()
    # example_2d_regression()
    # classification xor
    # example_classification_xor()
    # example_classification_xor_locally()
    # examples of sklearn
    classification = Classification()
    classification.common_examples(500, 30)
    # classification.common_examples_locally(M1=50, M2=50, swaps=5)

    # regression.common_examples(200, 30)
    # regression.common_examples_locally(M1=40, M2=40, swaps=5)


def main():
    check_models()
    # Make regression
    # make_regression()
    # make classification
    # Gaussian Learning
    # exampleGMM1D()
    # exampleGMM2D()
    print("stop")


def bank_locally_class(X, y, M1, M2, swaps, name_to_print, bias=1, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    del X, y
    bank_locally = BaNK_classification.bank_locally_classification(X_train, y_train, M1, M2, bias=bias)
    bank_locally.learn_kernel(swaps, X, y, name_to_print)
    print("End train ---------------------------" + name_to_print + "-------------------------Time: " + str(
        time.strftime("%c")) + "----------")
    score = str(bank_locally.score(X_test, y_test))
    print("score: " + name_to_print + " " + score + " swaps: " + str(swaps) + " M1: " + str(M1) + " M2: " + str(M2))
    print("Confussion matrix test set: ")
    print(bank_locally.return_confussion_matrix(X_test, y_test))
    del bank_locally, X_train, X_test, y_train, y_test


def bank_class(X, y, M, swaps, name_to_print, bias=1, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    del X, y
    bank = BaNK_classification.bank_classification(X_train, y_train, M, bias=bias)
    bank.learn_kernel(swaps, X, y, name_to_print)
    bank.learn_kernel(swaps, X, y, name_to_print)
    print("End train ---------------------------" + name_to_print + "-------------------------Time: " + str(
        time.strftime("%c")) + "----------")
    score = str(bank.score(X_test, y_test))
    print("score: " + name_to_print + " " + score + " swaps: " + str(swaps) + " M: " + str(M))
    print("Confussion matrix test set: ")
    print(bank.return_confussion_matrix(X_test, y_test))
    del bank, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # para Usar el localmente estacionario:
    # bank_locally_class(X, y, M1, M2, swaps, name_to_print)

    # Para usar el estacionario:
    # bank_class(X, y, M, swaps, name_to_print)
    main()

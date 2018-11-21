import BaNK_extended
import BaNK_classification
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.stats import beta
from scipy.stats import multivariate_normal
import printGMM
import infinite_GMM
import pandas as pd
import datetime
import gc


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

def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, means.shape[1]))
    U = np.random.uniform(0, 1, size=N)
    for i in range(N):
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > U[i]:
                xi = sc.random.multivariate_normal(means[index], cov[index], size=1)[0]
                sampling[i] = xi
                break
            index += 1

    return sampling

def phi_xi(x, w):
    argument = w.dot(x).T

    return 1./ np.sqrt(len(w)) * np.concatenate((np.cos(argument), np.sin(argument)))

def __matrix_phi(X, omegas):
        means = []
        for x in X:
            means.append(phi_xi(x, omegas))
        return np.array(means)


def f(X, omegas, beta):
    Phi_x = __matrix_phi(X, omegas)

    means = np.array(Phi_x).dot(beta)
    Y = []
    for u in means:
        Y.append(np.random.normal(u, 1, size=1)[0])
    # Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
    return np.array(Y)


def printKernel(X, means, sigmas, pik):
    Y = np.zeros_like(X)
    for i in range(len(means)):
        Y += pik[i]*np.exp(-1./2*(sigmas[i] * X**2))* np.cos(means[i]*X)
    return Y

def example_1D_regression():
    means = np.array([-4*np.pi/3, 10/4., 0])
    cov = np.array([1./4, 1./40, 1/4**2])
    realpik = np.array([1. / 3, 1. / 3, 1. / 3])
    N = 5000
    M = 350
    real_omegas = samplingGMM_1d(N=M, means=means, cov=cov, pi=realpik)
    real_beta = np.array(multivariate_normal.rvs(mean=np.zeros(2 * M), cov=np.identity(2 * M), size=1))
    Xi = np.linspace(-1000, 1000, N)
    Yi = f(Xi, real_omegas, real_beta)
    plt.scatter(Xi, Yi, label='Samples')
    plt.legend()
    plt.show()
    number_of_rounds = 40
    bank = BaNK_extended.bank_regression(Xi, Yi, M)
    bank.learn_kernel(number_of_rounds)
    bank.sample_beta_sigma()
    beta_sampled = bank.beta
    sigma_e = bank.sigma_e
    Yi = f(Xi, real_omegas, real_beta)
    real_Phi_X = __matrix_phi(Xi, real_omegas)
    Phi_X = __matrix_phi(Xi, bank.omegas)
    # Yi_predicted = np.random.multivariate_normal(Phi_X.dot(bank.beta), sigma_e * np.identity(len(Phi_X)))
    Yi_full_predicted = bank.predict_new_X(Xi)
    error = Phi_X.dot(bank.beta) - real_Phi_X.dot(real_beta)
    print("MSE 1D: " + str(np.abs(error).mean()))
    # plt.scatter(Xi, Yi, label='Real Yi')
    # plt.scatter(Xi, Yi_predicted, label='Predicted Yi')
    plt.plot(Xi, real_Phi_X.dot(real_beta), label='Real mean', color='black')
    plt.plot(Xi, real_Phi_X.dot(real_beta) + 3 * 1, '--', color='black', label='Real variance')
    plt.plot(Xi, real_Phi_X.dot(real_beta) - 3 * 1, '--', color='black')
    plt.plot(Xi, Phi_X.dot(beta_sampled), label='Sample mean', color='red')
    plt.plot(Xi, Phi_X.dot(beta_sampled) + 3 * np.sqrt(sigma_e), '--', color='red', label='Sampled variance')
    plt.plot(Xi, Phi_X.dot(beta_sampled) - 3 * np.sqrt(sigma_e), '--', color='red')
    plt.legend()
    plt.show()

    Xi = np.linspace(-1100, 1100, N)
    real_Phi_X = __matrix_phi(Xi, real_omegas)
    Phi_X = __matrix_phi(Xi, bank.omegas)

    plt.plot(Xi, real_Phi_X.dot(real_beta), label='Real mean', color='black')
    plt.plot(Xi, real_Phi_X.dot(real_beta) + 3 * 1, '--', color='black', label='Real variance')
    plt.plot(Xi, real_Phi_X.dot(real_beta) - 3 * 1, '--', color='black')
    plt.plot(Xi, Phi_X.dot(beta_sampled), label='Sample mean', color='red')
    plt.plot(Xi, Phi_X.dot(beta_sampled) + 3 * np.sqrt(sigma_e), '--', color='red', label='Sampled variance')
    plt.plot(Xi, Phi_X.dot(beta_sampled) - 3 * np.sqrt(sigma_e), '--', color='red')
    plt.legend()
    plt.show()

def example_2d_regression():
    means = np.array([[-1, 0], [3. * np.pi / 4, 11. * np.pi / 8]])
    cov = np.array([[[1/2., 0], [0, 1/3.]], [[1./4, 1./-5], [1./-5, 1./5.3]]])
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
    bank = BaNK_extended.bank(Xi, Yi, M, real_omegas)
    bank.learn_kernel(number_of_rounds)
    bank.sample_beta_sigma()
    beta_sampled = bank.beta
    sigma_e = bank.sigma_e
    Yi = f(Xi, real_omegas, real_beta)
    real_Phi_X = __matrix_phi(Xi, real_omegas)
    Phi_X = __matrix_phi(Xi, bank.omegas)
    Yi_predicted = np.random.multivariate_normal(Phi_X.dot(bank.beta), sigma_e * np.identity(len(Phi_X)))
    Yi_full_predicted = bank.predict_new_X(Xi)
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

def print_ggm_1D(real_parameters, computed_parameters, separatedGaussians = True):
    real_mean, real_cov, real_pik = real_parameters[0], np.sqrt(real_parameters[1]), real_parameters[2]
    mean_sampled, cov_sampled, pik_sampled = computed_parameters[0], np.sqrt(computed_parameters[1]), computed_parameters[2]


    #real parameters
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
            plt.plot(X, Y, label = 'Real Gaussian Mixture', color='b', linestyle='--')
        else:
            Y += pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)

    if not separatedGaussians:
        plt.plot(X, Y, label = 'Real Gaussian Mixture')

    #sample parameters
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
    means, variance, pik = np.array([-4, -50, 10]), np.array([4, 4, 4]), np.array([1./3, 1./3, 1./3])
    X = samplingGMM_1d(500, means, np.sqrt(variance), pik)
    inf_learn = infinite_GMM.infinite_GMM(X, initial_number_class=3)
    inf_learn.learn_GMM(600)
    print_ggm_1D(np.array([means, variance, pik]), np.array([inf_learn.means, 1./inf_learn.S, inf_learn.get_weights()]))
    print_ggm_1D(np.array([means, variance, pik]), np.array([inf_learn.means, 1. / inf_learn.S, inf_learn.get_weights()]), False)
    pltgmm = printGMM.plotgaussianmixture(X, inf_learn.means, np.sqrt(1./inf_learn.S), inf_learn.get_weights())
    pltgmm_real = printGMM.plotgaussianmixture(X, means, np.sqrt(variance), pik)
    pltgmm.print_seperategaussians()
    pltgmm_real.print_seperategaussians()
    pltgmm.printGMM()
    pltgmm_real.printGMM()
    print ("some")

def exampleGMM2D():
    means, variance, pik = np.array([[0, 1], [20, 60], [100, 100], [10,10]]), np.array([[[4,0], [0,2]], [[4,0], [0,2]], [[4,0], [0,2]], [[4,0], [0,2]]]), np.array([1. / 4, 1. / 4, 1. / 4, 1./4])
    X = samplingGMM(500, means, variance, pik)
    inf_learn = infinite_GMM.infinite_GMM(X, initial_number_class=3)
    inf_learn.learn_GMM(600)
    pltgmm = printGMM.plotgaussianmixture(X, inf_learn.means, inf_learn.get_covariance_matrix(), inf_learn.get_weights())
    pltgmm.print_seperategaussians()
    pltgmm.printGMM()
    print ("some")


def example_classification_xor():
    size1 = 1000
    means = np.array([[0, 0], [10, 10]])
    covarance = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1, means, covarance, [1./2, 1./2])

    means = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1, means, covarance, [1./2, 1./2])
    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))

    plt.scatter(X1.T[0], X1.T[1], label = 'Class 1')
    plt.scatter(X2.T[0], X2.T[1], label = 'Class 2')
    plt.legend()
    plt.show()

    bank_cl = BaNK_classification.bank_classification(X, Y, 250)
    bank_cl.learn_kernel(5)

    means = np.array([[0, 0], [10, 10]])
    covarance = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X1 = samplingGMM(size1*10, means, covarance, [1. / 2, 1. / 2])

    means = np.array([[0, 10], [10, 0]])
    X2 = samplingGMM(size1*10, means, covarance, [1. / 2, 1. / 2])

    Y = np.append(np.tile(1, len(X2)), np.zeros(len(X1)))
    X = np.vstack((X1, X2))
    X = np.column_stack((X, Y))

    np.random.shuffle(X)

    #Ver si esto funciona
    Y_predict = bank_cl.predict_new_X(X.T[:2].T)

    z = np.where(X.T[2] == 0)
    plt.scatter(X[z].T[0], X[z].T[1], label='Class 1')

    z = np.where(X.T[2] == 1)
    plt.scatter(X[z].T[0], X[z].T[1], label='Class 2')
    plt.legend()
    plt.show()

    error = np.abs(Y_predict - X.T[2])

    print("Error: " + str(int(np.sum(error))) + " in " + str(len(error)))


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

    print("Error: " + str(np.sum(error)/len(y)))

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
    n_train = int(n*0.8)

    for i in range(times_crossvalidation):
        time_begin = datetime.datetime.now()
        print("Begin " + str(i+1) + " of " + str(times_crossvalidation) + " at " + str(time_begin.strftime("%A, %d. %B %Y %I:%M%p")))

        np.random.shuffle(X)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        bank = BaNK_classification.bank_classification(X_train, Y_train, M)
        gc.collect()
        bank.learn_kernel(number_rounds)
        time_end = datetime.datetime.now()
        print("End " + str(i+1) + " of " + str(times_crossvalidation) + " at " + str(time_end.strftime("%A, %d. %B %Y %I:%M%p")))
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

    df = pd.read_csv('dataset regression/skin segmentation.csv', sep = '\t')
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
    n_train = int(n*0.8)
    rangey = max(Y) - min(Y)

    for i in range(times_crossvalidation):
        time_begin = datetime.datetime.now()
        print("Begin " + str(i+1) + " of " + str(times_crossvalidation) + " at " + str(time_begin.strftime("%A, %d. %B %Y %I:%M%p")))

        np.random.shuffle(X)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        bank = BaNK_extended.bank_regression(X_train, Y_train, M)
        bank.learn_kernel(number_rounds)
        time_end = datetime.datetime.now()
        print("End " + str(i+1) + " of " + str(times_crossvalidation) + " at " + str(time_end.strftime("%A, %d. %B %Y %I:%M%p")))
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
    M = 250
    df = pd.read_csv('dataset regression/Airfoil Self-Noise Data Set.csv', sep='\t')
    X1 = np.array(df.T[:5].T)
    Y1 = np.array(df['output (Db)'])

    number_of_rounds = 30
    scores_airfoil = cross_validation_regression(X1, Y1, M, number_of_rounds, 5)
    print("Scores of Airfoil with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    print("Range: " + str(max(Y1) - min(Y1)))
    for row in scores_airfoil:
        print("MSE Airfoil: " + str(row[0]) + " +- " + str(row[1]))
    print("----------------------------------------")
    #bike dataset
    # M = 750
    df = pd.read_csv('dataset regression/bike.csv')
    X2 = np.array(df.T[:18].T)
    Y2 = np.array(df['cnt'])

    number_of_rounds = 15
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

    number_of_rounds = 15
    scores_airfoil = cross_validation_regression(X3, Y3, M, number_of_rounds, 5)
    print("Scores of concrete with " + str(M) + " angles and " + str(number_of_rounds) + " swaps")
    print("Range: " + str(max(Y3) - min(Y3)))
    for row in scores_airfoil:
        print("MSE concrete: "  + str(row[0]) + " +- " + str(row[1]))
    print("----------------------------------------")
    print("Finish regression")

def check_models():
    #regression
    example_1D_regression()
    example_2d_regression()
    #classification xor
    example_classification_xor()

def __main():
    check_models()
    #Make regression
    # make_regression()
    #make classification
    # make_classification()
    #Clasifiaction example
    # example_2_spiral()
    example_classification_xor()
    #Gaussian Learning
    # exampleGMM1D()
    exampleGMM2D()
    #Kernel learning
    example_1D()
    example_2d()
    print ("stop")


if __name__ == '__main__':
    __main()
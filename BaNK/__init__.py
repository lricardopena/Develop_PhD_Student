import BaNK_extended
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from scipy.stats import invwishart, invgamma
from scipy.special import gamma
from scipy.special import gammaln
from scipy.stats import norm
import printGMM


def samplingGMM(N, means, cov, pi):
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

def phi_xi(x, w):
    argument = w.dot(x).T

    return np.concatenate(([1], np.cos(argument), np.sin(argument)))


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

def __main():
    means, cov, realpik = np.array([0, 3. * np.pi / 4]), np.array([1.0 / 2 ** 2, 1.0 / 2 ** 2]), np.array([1. / 2, 1. / 2])
    N = 1000
    M = 250
    Xi = norm.rvs(loc=0, scale=4, size=N)
    real_omegas = samplingGMM(N=M, means=means, cov=np.sqrt(cov), pi=realpik)
    real_beta = np.array(multivariate_normal.rvs(mean=np.zeros(2 * M+1), cov=np.identity(2 * M+1), size=1))
    # beta = np.concatenate(([1], beta))
    Yi = f(Xi, real_omegas, real_beta)
    #
    # # Xi = np.linspace(-7, 7, 1000)
    # # Yi_real = printKernel(Xi, means, cov, realpik)
    # # plt.plot(Xi, Yi_real)
    # # plt.show()


    real_Phi_x = __matrix_phi(Xi, real_omegas)


    number_of_rounds = 2000


    bank = BaNK_extended.bank(Xi, Yi, M, real_omegas)
    Phi_X_computed = bank.get_Phi_X(Xi, real_omegas)
    #
    Yi_predict = bank.predict_new_X(Xi, real_omegas)
    plt.scatter(Xi, Yi, label='Real Data')
    plt.scatter(Xi, Yi_predict, label='Predicted data')
    plt.legend()
    plt.show()
    bank.learn_kernel(number_of_rounds)
    Xi = np.linspace(-10, 10, 1000)
    plt.plot(Xi, printKernel(Xi, bank.means, np.sqrt(1./bank.S), bank.get_pik()), label='kernel learning')
    plt.plot(Xi, printKernel(Xi, means, cov, realpik), label='True kernel')
    # plt.plot(Xi, BaNKKernel(Xi, mus, sigmas, pik_get), label='BaNK')
    plt.legend()
    plt.show()

    new_size = 1000
    Xi_new = norm.rvs(loc=0, scale=4, size=new_size)

    Yi_new = f(Xi_new, real_omegas, real_beta)
    Yi_predicted = bank.predict_new_X(Xi_new, bank.omegas)

    plt.scatter(Xi_new, Yi_new, label='real Yi')
    plt.scatter(Xi_new, Yi_predicted, label='predicted')
    plt.legend()
    plt.show()
    print ("Error: " + str(1./new_size*np.sum(np.abs(Yi_new-Yi_predicted))))
    print ("stop")

if __name__ == '__main__':
    __main()
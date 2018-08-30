import numpy as np
import scipy as sc
import scipy.stats
import matplotlib.pyplot as plt


class learn_gmm:
    def __init__(self, X, alpha, TD, lamda_plus_min_global = 0.95):
        self.X = X
        self.alpha = alpha
        self.TD = TD
        self.lamda_plus_min_global = lamda_plus_min_global
        N = len(X)
        self.lamda_minus_min_global = np.exp(N * np.log(self.lamda_plus_min_global)/sc.stats.chi2.sf(alpha, df=N-1, loc=alpha))
        self.TD_computed = N / sc.stats.chi2.sf(alpha, df=N - 1, loc=alpha) * TD ** 2
        self.Z = np.zeros(N)

    def __compute_lambda(self, Xk, mean, cov):
        D = self.__kolmogorov_smirnoff_test(Xk, mean, cov)
        return np.exp(-(D**2/self.TD**2))

    def __kolmogorov_smirnoff_test(self, Xk, mean, cov):
        dimension = mean.shape[0]
        D = np.zeros(dimension)
        N = len(Xk)
        for d in range(dimension):
            meand = mean[d]
            variance = np.sqrt(cov[d][d])
            Xd = np.array(np.sort(Xk.T[d]))
            cdf_d_dimension = sc.stats.norm.cdf(Xd, loc=meand, scale=variance)
            Y = np.array(range(N))/float(N-1)
            D[d] = np.sum(np.absolute(cdf_d_dimension - Y))/float(N)
            plt.plot(Xd, cdf_d_dimension, label="Cumulative")
            plt.plot(Xd, Y, label='Empirical Cumulative')
            plt.legend()
            plt.show()

        return np.sum(D)

    # def __combine_two_gaussians(self,i, j):

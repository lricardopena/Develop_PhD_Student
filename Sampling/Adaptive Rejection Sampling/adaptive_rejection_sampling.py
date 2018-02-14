'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Gilks, W. R., & Wild, P. (1992). Adaptive rejection sampling for Gibbs sampling. Applied Statistics, 337-348.

It is necessary to give and h function, which is log concave and its derivative as parameters including the limits inferior and superior
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats


class ARS:

    def __init__(self, Tk, x0, xk_plus_1, number_of_elements_linespace, h_function, h_derivative_function, *args):
        self.args = args
        self.Tk = Tk
        self.x0 = x0
        self.xk_plus_1 = xk_plus_1
        self.number_of_elements_linespace = number_of_elements_linespace
        self.h_function = h_function
        self.h_derivative_function = h_derivative_function

    def compute_all_Zj(self):
        '''
        :return: A list with all z1, z2, ...., zk (all the intersection between points)
        '''
        Z = []
        for j in range(-1, len(self.Tk)):
            zj = self.compute_Zj(j)
            Z.append(zj)
        return Z

    def compute_Zj(self, j):
        '''
        :param j: index for Tk
        :return: The element zj, where interect xj and xj_plus_1
        '''
        if j == -1:
            xj = self.x0
            xj_plus_1 = self.Tk[0]

        elif j < len(self.Tk) - 1:
            xj = self.Tk[j]
            xj_plus_1 = self.Tk[j + 1]
        else:
            xj = self.Tk[j]
            xj_plus_1 = self.xk_plus_1

        zj = (self.h_function(xj_plus_1, *self.args) - self.h_function(xj,
                                                                       *self.args) - xj_plus_1 * self.h_derivative_function(
            xj_plus_1, *self.args) + xj * self.h_derivative_function(xj, *self.args)) / (
                     self.h_derivative_function(xj, *self.args) - self.h_derivative_function(xj_plus_1, *self.args))

        return zj

    def compute_lk_single_x(self, x):
        '''
        :param x: Is variable to compute the value of l_k(x)
        :return: The value of l_k(x)
        '''
        if x < self.x0:
            x = self.x0
        elif x > self.xk_plus_1:
            x = self.xk_plus_1

        if x < self.Tk[0]:
            xj = self.x0
            xj_plus_1 = self.Tk[0]
        elif x > self.Tk[len(self.Tk) - 1]:
            xj = self.Tk[len(self.Tk) - 1]
            xj_plus_1 = self.xk_plus_1
        else:
            j = np.where(self.Tk >= x)[0][0]
            xj = self.Tk[j - 1]
            xj_plus_1 = self.Tk[j]

        return ((xj_plus_1 - x) * self.h_function(xj, *self.args) + (x - xj) * self.h_function(xj_plus_1,
                                                                                               *self.args)) / (
                       xj_plus_1 - xj)

    def compute_lk(self):
        '''
        :return: All DISCRETE values of l_k(x) and all x related to x value of l_k(x)
        '''
        lk = np.array([])
        Xk = np.array([])

        for j in range(-1, len(self.Tk)):
            if j < 0:  # Start x0
                xj = self.x0
                xj_plus_1 = self.Tk[0]
            elif j < len(self.Tk) - 1:
                xj = self.Tk[j]
                xj_plus_1 = self.Tk[j + 1]
            else:  # Finish xk_plus_1
                xj = self.Tk[j]
                xj_plus_1 = self.xk_plus_1

            X = np.linspace(xj, xj_plus_1)
            Xk = np.append(Xk, X)
            Y = ((xj_plus_1 - X) * self.h_function(xj, *self.args) + (X - xj) * self.h_function(xj_plus_1,
                                                                                                *self.args)) / (
                        xj_plus_1 - xj)
            lk = np.append(lk, Y)

        return lk, Xk

    def compute_uk_single_x(self, x, Z):
        '''
        :param x: Is variable to compute the value of u_k(x)
        :param Z: Is the list with all values in z1, z2,... zk (the intersection between xj and xj_plus_1
        :return: The result of compute u_k(x)
        '''
        # we truncate x (the samples) between x0 and xk_plus_1
        if x < self.x0:
            x = self.x0
        elif x > self.xk_plus_1:
            x = self.xk_plus_1

        if x > Z[len(Z) - 1]:
            xj = self.xk_plus_1
        elif x < Z[0]:
            xj = self.x0
        else:
            j = np.where(Z >= x)[0][0]
            # Z[j-1] <= x <= Z[j]
            xj = self.Tk[j - 1]
        # (h_log_normal(xj, mean, sigma)) + (x - xj) * h_derivative_log_normal(xj, mean, sigma)
        return (self.h_function(xj, *self.args)) + (x - xj) * self.h_derivative_function(xj, *self.args)

    def compute_uk(self, Z):
        '''
        :param List Z: Is the list with all values in z1, z2,... zk (the intersection between xj and xj_plus_1
        :return: All the DISCRETE points in u_k(x) and its correspondent x between x0 and x_plus_1
        '''
        uk = np.array([])
        Xk = np.array([])

        for j in range(-1, len(Z)):
            if j < 0:  # Start x0
                xj = self.x0
                X = np.linspace(self.x0, Z[0], self.number_of_elements_linespace)
            elif j < len(self.Tk):
                xj = self.Tk[j]
                X = np.linspace(Z[j], Z[j + 1], self.number_of_elements_linespace)
            else:  # finish xk_plus_1
                xj = self.xk_plus_1
                X = np.linspace(Z[j], self.xk_plus_1, self.number_of_elements_linespace)
            Xk = np.append(Xk, X)
            Y = (self.h_function(xj, *self.args)) + (X - xj) * self.h_derivative_function(xj, *self.args)
            uk = np.append(uk, Y)

        return uk, Xk

    def sample_from_sk(self, sk, Xk, plot_every_step):
        '''
        :param sk: All elements in s_k(x)
        :param Xk: The correspondendt value x to s_k(x)
        :param plot_every_step: Boolean to determine if plot every sample from s_k(x)
        :return: a sample from from s_k(x)
        '''
        cumulative_sk = np.cumsum(sk / np.sum(sk))

        u = np.random.uniform()
        indexXj = np.where(cumulative_sk >= u)[0][0]

        x = Xk[indexXj]
        if plot_every_step:
            plt.plot(Xk, cumulative_sk, label='Cumulate')
            plt.plot([Xk[0], x, x], [u, u, 0], label='x sampled: ' + str(np.round(x, 3)))
            Y = []
            for j, xk in enumerate(self.Tk):
                Y.append(cumulative_sk[np.where(Xk >= xk)][0])
            plt.scatter(self.Tk, Y, label='Elements in Tk')
            plt.legend()
            plt.show()
        return x

    def compute_sk(self, Zk):
        '''
        :return: An array with all elements DESCRETE from x0 to xk_plus_1 in s_k(x) and the corresponding values x
        '''
        Uk, Xk = self.compute_uk(Zk)
        sk = np.exp(Uk)
        return sk, Xk

    def perform_ARS_log_normal(self, number_of_samples, plot_every_step):
        '''
        :param number_of_samples: The number of samples to be generated
        :param number_of_elements_linespace: The number of elements between xj and xj_plus_1 (Big numbers of elements make
        :param plot_every_step: Boolean True if you want to plot every step in ARS
        :return: Samples generated from h(x)
        '''
        x_samples = []
        Z = self.compute_all_Zj()
        Sk, Xk = self.compute_sk(Z)

        while len(x_samples) < number_of_samples:
            x_proposal = self.sample_from_sk(Sk, Xk, plot_every_step)

            lk_x_proposal = self.compute_lk_single_x(x_proposal)
            uk_x_proposal = self.compute_uk_single_x(x_proposal, Z)

            p = np.exp(lk_x_proposal - uk_x_proposal)
            w = np.random.uniform()
            if p > w:
                x_samples.append(x_proposal)
            else:
                self.Tk.append(x_proposal)
                self.Tk = sorted(self.Tk)
                Z = self.compute_all_Zj()
                Sk, Xk = self.compute_sk(Z)

            if plot_every_step:
                X = np.linspace(self.x0, self.xk_plus_1, 1000)
                Y = self.h_function(X, *self.args)
                plt.plot(X, Y, label='h(x)')

                Tk_ext = np.array([self.x0])
                Tk_ext = np.append(Tk_ext, self.Tk)
                Tk_ext = np.append(Tk_ext, [self.xk_plus_1])
                Y = self.h_function(Tk_ext, *self.args)
                plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

                Z = self.compute_all_Zj()
                Uk = np.log(Sk)
                plt.plot(Xk, Uk, label='Lines Uk')
                Lk, Xk = self.compute_lk()
                plt.plot(Xk, Lk, label='Lines lk')

                if len(x_samples) > 0:
                    Y = []
                    for x_samp in x_samples:
                        y = self.compute_uk_single_x(x_samp)
                        Y.append(y)
                    plt.scatter(x_samples, Y, label='Samples')

                plt.legend()
                plt.show()

        if plot_every_step:
            X = np.linspace(self.x0, self.xk_plus_1, 1000)
            Y = self.h_function(X, *self.args)
            plt.plot(X, Y, label='h(x)')

            Tk_ext = np.array([self.x0])
            Tk_ext = np.append(Tk_ext, self.Tk)
            Tk_ext = np.append(Tk_ext, [self.xk_plus_1])
            Y = self.h_function(Tk_ext, *self.args)
            plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

            Z = self.compute_all_Zj()
            Uk, Xk = self.compute_uk(Z)
            plt.plot(Xk, Uk, label='Lines Uk')
            Lk, Xk = self.compute_lk()
            plt.plot(Xk, Lk, label='Lines lk')

            if len(x_samples) > 0:
                Y = []
                for x_samp in x_samples:
                    y = self.compute_uk_single_x(x_samp, Z)
                    Y.append(y)
                plt.scatter(x_samples, Y, label='Samples')

            plt.legend()
            plt.show()

        return x_samples


def h_log_normal(X, *args):
    mean = args[0]
    sigma = args[1]
    return -((X - mean)) ** 2 / (2 * sigma ** 2)


def h_derivative_log_normal(X, *args):
    mean = args[0]
    sigma = args[1]
    return -(X - mean) / (sigma ** 2)

def main():
    mean = 100.
    sigma = 10.
    Tk = [-30, 170]
    x0 = -500
    xk_plus_1 = 700

    ars_Sampling = ARS(Tk, x0, xk_plus_1, 10000, h_log_normal, h_derivative_log_normal, mean, sigma)

    x_samples = ars_Sampling.perform_ARS_log_normal(10000, False)

    x_left = np.min(x_samples)
    x_right = np.max(x_samples)
    X = np.linspace(x_left, x_right, 1000)
    Y = sc.stats.norm.pdf(X, loc=mean, scale=sigma)
    plt.plot(X, Y)
    plt.hist(x_samples, bins='auto', normed=True, label='Samples generated by ARS')

    # plt.hist(x_samples, bins='auto', density=True)

    print "mean: " + str(np.mean(x_samples))
    print "sigma: " + str(np.std(x_samples))
    plt.show()


if __name__ == "__main__":
    main()

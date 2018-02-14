'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Gilks, W. R., & Wild, P. (1992). Adaptive rejection sampling for Gibbs sampling. Applied Statistics, 337-348.

It is necessary to give and h function, which is log concave and its derivative as parameters including the limits inferior and superior
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats


def compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args):
    '''
    :param List Tk: List of with all rejected sampled
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return: A list with all z1, z2, ...., zk (all the intersection between points)
    '''
    Z = []
    for j in range(-1, len(Tk)):
        zj = compute_Zj(j, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
        Z.append(zj)
    return Z


def compute_Zj(j, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args):
    '''
    :param j: index for Tk
    :param Tk: List of with all rejected sampled
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return: The element zj, where interect xj and xj_plus_1
    '''
    if j == -1:
        xj = x0
        xj_plus_1 = Tk[0]

    elif j < len(Tk) - 1:
        xj = Tk[j]
        xj_plus_1 = Tk[j + 1]
    else:
        xj = Tk[j]
        xj_plus_1 = xk_plus_1

    zj = (h_function(xj_plus_1, *args) - h_function(xj, *args) - xj_plus_1 * h_derivative_function(xj_plus_1,
                                                                                                   *args) + xj * h_derivative_function(
        xj, *args)) / (h_derivative_function(xj,
                                             *args) - h_derivative_function(xj_plus_1, *args))

    return zj


def compute_lk_single_x(x, Tk, x0, xk_plus_1, h_function, *args):
    '''
    :param x: Is variable to compute the value of l_k(x)
    :param Tk: All sampled rejected
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param args: Is all the arguments needed to the h function and its derivative
    :return: The value of l_k(x)
    '''
    if x < x0:
        x = x0
    elif x > xk_plus_1:
        x = xk_plus_1

    if x < Tk[0]:
        xj = x0
        xj_plus_1 = Tk[0]
    elif x > Tk[len(Tk) - 1]:
        xj = Tk[len(Tk) - 1]
        xj_plus_1 = xk_plus_1
    else:
        j = np.where(Tk >= x)[0][0]
        xj = Tk[j - 1]
        xj_plus_1 = Tk[j]

    return ((xj_plus_1 - x) * h_function(xj, *args) + (x - xj) * h_function(xj_plus_1, *args)) / (
                           xj_plus_1 - xj)


def compute_lk(Tk, x0, xk_plus_1, h_function, *args):
    '''
    :param Tk: All sampled rejected
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param args: Is all the arguments needed to the h function and its derivative
    :return: All DISCRETE values of l_k(x) and all x related to x value of l_k(x)
    '''
    lk = np.array([])
    Xk = np.array([])

    for j in range(-1, len(Tk)):
        if j < 0:  # Start x0
            xj = x0
            xj_plus_1 = Tk[0]
        elif j < len(Tk) - 1:
            xj = Tk[j]
            xj_plus_1 = Tk[j + 1]
        else:  # Finish xk_plus_1
            xj = Tk[j]
            xj_plus_1 = xk_plus_1

        X = np.linspace(xj, xj_plus_1)
        Xk = np.append(Xk, X)
        Y = ((xj_plus_1 - X) * h_function(xj, *args) + (X - xj) * h_function(xj_plus_1, *args)) / (
                xj_plus_1 - xj)
        lk = np.append(lk, Y)

    return lk, Xk


def compute_uk_single_x(x, Z, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args):
    '''
    :param x: Is variable to compute the value of u_k(x)
    :param Z: Is the list with all values in z1, z2,... zk (the intersection between xj and xj_plus_1
    :param Tk: All sampled rejected
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return:
    '''
    # we truncate x (the samples) between x0 and xk_plus_1
    if x < x0:
        x = x0
    elif x > xk_plus_1:
        x = xk_plus_1

    if x > Z[len(Z) - 1]:
        xj = xk_plus_1
    elif x < Z[0]:
        xj = x0
    else:
        j = np.where(Z >= x)[0][0]
        # Z[j-1] <= x <= Z[j]
        xj = Tk[j - 1]
    # (h_log_normal(xj, mean, sigma)) + (x - xj) * h_derivative_log_normal(xj, mean, sigma)
    return (h_function(xj, *args)) + (x - xj) * h_derivative_function(xj, *args)


def compute_uk(Z, Tk, x0, xk_plus_1, number_of_elements_linespace, h_function, h_derivative_function, *args):
    '''
    :param List Z: Is the list with all values in z1, z2,... zk (the intersection between xj and xj_plus_1
    :param Tk: All sampled rejected
    :param x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param number_of_elements_linespace: The number of elements between zj and zj_plus_1
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return: All the DISCRETE points in u_k(x) and its correspondent x between x0 and x_plus_1
    '''
    uk = np.array([])
    Xk = np.array([])

    for j in range(-1, len(Z)):
        if j < 0:  # Start x0
            xj = x0
            X = np.linspace(x0, Z[0], number_of_elements_linespace)
        elif j < len(Tk):
            xj = Tk[j]
            X = np.linspace(Z[j], Z[j + 1], number_of_elements_linespace)
        else:  # finish xk_plus_1
            xj = xk_plus_1
            X = np.linspace(Z[j], xk_plus_1, number_of_elements_linespace)
        Xk = np.append(Xk, X)
        Y = (h_function(xj, *args)) + (X - xj) * h_derivative_function(xj, *args)
        uk = np.append(uk, Y)

    return uk, Xk


def sample_from_sk(sk, Xk, plot_every_step, Tk):
    '''
    :param sk: All elements in s_k(x)
    :param Xk: The correspondendt value x to s_k(x)
    :param plot_every_step: Boolean to determine if plot every sample from s_k(x)
    :param Tk: A list with rejected samples
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
        for j, xk in enumerate(Tk):
            Y.append(cumulative_sk[np.where(Xk >= xk)][0])
        plt.scatter(Tk, Y, label='Elements in Tk')
        plt.legend()
        plt.show()
    return x


def compute_sk_sample_x(Tk, x0, xk_plus_1, number_of_elements_linespace, h_function, h_derivative_function, *args):
    '''
    :param List Tk: All sampled rejected
    :param float x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param number_of_elements_linespace: The number of elements between zj and zj_plus_1
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return: An array with all elements DESCRETE from x0 to xk_plus_1 in s_k(x) and the corresponding values x
    '''
    Z = compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
    Uk, Xk = compute_uk(Z, Tk, x0, xk_plus_1, number_of_elements_linespace, h_function, h_derivative_function, *args)
    sk = np.exp(Uk)
    return sk, Xk


def perform_ARS_log_normal(number_of_samples, Tk, x0, xk_plus_1, plot_every_step, number_of_elements_linespace,
                           h_function, h_derivative_function, *args):
    '''
    :param number_of_samples: The number of samples needed
    :param Tk: Initial elements in rejected samples
    :param float x0: Left bound to the sampling
    :param xk_plus_1: Right bound to the sampling
    :param plot_every_step: Boolean True if you want to plot every step in ARS
    :param number_of_elements_linespace: The number of elements between xj and xj_plus_1 (Big numbers of elements make
    the sampling more accurate, but more slow)
    :param h_function: Function h to be log concave, the arguments of the function must be like h(x, *args), where x is
    the sample and *args is all other arguments
    :param h_derivative_function: Derivative of h function, it must be like h'(x, *args), where x is the sample and
    *args is all other arguments necessary to the function
    :param args: Is all the arguments needed to the h function and its derivative
    :return: Samples generated from h(x), and all rejected sampled (Tk)
    '''
    x_samples = []
    Zk = compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)

    Sk, Xk = compute_sk_sample_x(Tk, x0, xk_plus_1, number_of_elements_linespace, h_function,
                                 h_derivative_function, *args)
    while len(x_samples) < number_of_samples:

        x_proposal = sample_from_sk(Sk, Xk, plot_every_step, Tk)

        lk_x_proposal = compute_lk_single_x(x_proposal, Tk, x0, xk_plus_1, h_function, *args)
        uk_x_proposal = compute_uk_single_x(x_proposal, Zk, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)

        p = np.exp(lk_x_proposal - uk_x_proposal)
        w = np.random.uniform()
        if p > w:
            x_samples.append(x_proposal)
        else:
            Tk.append(x_proposal)
            Tk = sorted(Tk)
            Zk = compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
            Sk, Xk = compute_sk_sample_x(Tk, x0, xk_plus_1, number_of_elements_linespace, h_function,
                                         h_derivative_function, *args)

        if plot_every_step:
            X = np.linspace(x0, xk_plus_1, 1000)
            Y = h_function(X, *args)
            plt.plot(X, Y, label='h(x)')

            Tk_ext = np.array([x0])
            Tk_ext = np.append(Tk_ext, Tk)
            Tk_ext = np.append(Tk_ext, [xk_plus_1])
            Y = h_function(Tk_ext, *args)
            plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

            Z = compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
            Uk = np.log(Sk)
            plt.plot(Xk, Uk, label='Lines Uk')
            Lk, Xk = compute_lk(Tk, x0, xk_plus_1, h_function, *args)
            plt.plot(Xk, Lk, label='Lines lk')

            if len(x_samples) > 0:
                Y = []
                for x_samp in x_samples:
                    y = compute_uk_single_x(x_samp, Z, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
                    Y.append(y)
                plt.scatter(x_samples, Y, label='Samples')

            plt.legend()
            plt.show()

    if plot_every_step:
        X = np.linspace(x0, xk_plus_1, 1000)
        Y = h_function(X, *args)
        plt.plot(X, Y, label='h(x)')

        Tk_ext = np.array([x0])
        Tk_ext = np.append(Tk_ext, Tk)
        Tk_ext = np.append(Tk_ext, [xk_plus_1])
        Y = h_function(Tk_ext, *args)
        plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

        Z = compute_all_Zj(Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
        Uk, Xk = compute_uk(Z, Tk, x0, xk_plus_1, number_of_elements_linespace, h_function, h_derivative_function,
                            *args)
        plt.plot(Xk, Uk, label='Lines Uk')
        Lk, Xk = compute_lk(Tk, x0, xk_plus_1, h_function, *args)
        plt.plot(Xk, Lk, label='Lines lk')

        if len(x_samples) > 0:
            Y = []
            for x_samp in x_samples:
                y = compute_uk_single_x(x_samp, Z, Tk, x0, xk_plus_1, h_function, h_derivative_function, *args)
                Y.append(y)
            plt.scatter(x_samples, Y, label='Samples')

        plt.legend()
        plt.show()

    return x_samples, Tk


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

    x_samples, x_rejected = perform_ARS_log_normal(10000, Tk, x0, xk_plus_1, False, 10000, h_log_normal,
                                                   h_derivative_log_normal,
                                                   mean, sigma)

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

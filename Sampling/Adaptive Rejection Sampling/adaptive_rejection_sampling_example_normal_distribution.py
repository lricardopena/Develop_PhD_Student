'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Gilks, W. R., & Wild, P. (1992). Adaptive rejection sampling for Gibbs sampling. Applied Statistics, 337-348.
With only one example, log normal as h(x).
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.special
import scipy.stats


def h_log_normal(X, mean, sigma):
    return -((X - mean)) ** 2 / (2 * sigma ** 2)


def h_derivative_log_normal(X, mean, sigma):
    return -(X - mean) / (sigma ** 2)


def compute_all_Zj(Tk, mean, sigma, x0, xk_plus_1):
    Z = []
    for j in range(-1, len(Tk)):
        zj = compute_Zj(j, Tk, mean, sigma, x0, xk_plus_1)
        Z.append(zj)
    return Z


def compute_Zj(j, Tk, mean, sigma, x0, xk_plus_1):
    if j == -1:
        xj = x0
        xj_plus_1 = Tk[0]

    elif j < len(Tk) - 1:
        xj = Tk[j]
        xj_plus_1 = Tk[j + 1]
    else:
        xj = Tk[j]
        xj_plus_1 = xk_plus_1
    zj = (h_log_normal(xj_plus_1, mean, sigma) - h_log_normal(xj, mean,
                                                              sigma) - xj_plus_1 * h_derivative_log_normal(
        xj_plus_1, mean, sigma) + xj * h_derivative_log_normal(xj, mean, sigma)) / (
                 h_derivative_log_normal(xj, mean, sigma) - h_derivative_log_normal(xj_plus_1, mean, sigma))

    return zj


def compute_lk_single_x(x, mean, sigma, Tk, x0, xk_plus_1):
    if x < x0:
        x = x0
    elif x > xk_plus_1:
        x = xk_plus_1

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
        if xj <= x <= xj_plus_1:
            return ((xj_plus_1 - x) * h_log_normal(xj, mean, sigma) + (x - xj) * h_log_normal(xj_plus_1, mean,
                                                                                              sigma)) / (
                           xj_plus_1 - xj)


def compute_lk(mean, sigma, Tk, x0, xk_plus_1):
    lk = np.array([])
    Xk = np.array([])

    '''X = np.linspace(x0, xk_plus_1, 1000)
    Y = h_log_normal(X, mean, sigma)
    plt.plot(X, Y, label='h(x)')

    Tk_ext = np.array([x0])
    Tk_ext = np.append(Tk_ext, Tk)
    Tk_ext = np.append(Tk_ext, [xk_plus_1])
    Y = h_log_normal(Tk_ext, mean, sigma)
    plt.scatter(Tk_ext, Y, label='Elemnts in Tk')'''

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
        Y = ((xj_plus_1 - X) * h_log_normal(xj, mean, sigma) + (X - xj) * h_log_normal(xj_plus_1, mean, sigma)) / (
                xj_plus_1 - xj)
        lk = np.append(lk, Y)
        # plt.plot(X, Y, label='l_k(x' + str(j + 1) + ')')

    # plt.legend()
    # plt.show()
    return lk, Xk


def compute_uk_single_x(x, Z, mean, sigma, Tk, x0, xk_plus_1):
    if x < x0:
        x = x0
    elif x > xk_plus_1:
        x = xk_plus_1

    # we truncate the samples
    for j in range(-1, len(Z)):
        if j < 0:
            zj = x0
            zj_plus_1 = Z[j + 1]
            xj = x0
        elif j < len(Z) - 1:
            zj = Z[j]
            zj_plus_1 = Z[j + 1]
            xj = Tk[j]
        else:
            zj = Z[j]
            zj_plus_1 = xk_plus_1
            xj = xk_plus_1

        if zj <= x <= zj_plus_1:
            return (h_log_normal(xj, mean, sigma)) + (x - xj) * h_derivative_log_normal(xj, mean, sigma)
    print("error")


def compute_uk(Z, mean, sigma, Tk, x0, xk_plus_1, number_of_elements_linespace):
    uk = np.array([])
    Xk = np.array([])

    '''X = np.linspace(x0, xk_plus_1, 1000)
    Y = h_log_normal(X, mean, sigma)
    plt.plot(X, Y, label='h(x)')

    Tk_ext = np.array([x0])
    Tk_ext = np.append(Tk_ext, Tk)
    Tk_ext = np.append(Tk_ext, [xk_plus_1])
    Y = h_log_normal(Tk_ext, mean, sigma)
    plt.scatter(Tk_ext, Y, label='Elemnts in Tk')'''

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
        Y = (h_log_normal(xj, mean, sigma)) + (X - xj) * h_derivative_log_normal(xj, mean, sigma)
        uk = np.append(uk, Y)
        # plt.plot(X, Y, label='g_u(x' + str(j + 1) + ')')

    # plt.legend()
    # plt.show()

    return uk, Xk


def sample_from_sk(sk, Xk, plot_every_step, Tk):
    C = np.sum(sk)
    sk = sk / C
    cumulative_sk = np.cumsum(sk)



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
        # x = (np.log(u) - h_log_normal(xj, mean, sigma)) / h_derivative_log_normal(xj, mean, sigma) + xj
        plt.legend()
        plt.show()
    return x


def compute_sk_sample_x(Tk, mean, sigma, x0, xk_plus_1, number_of_elements_linespace):
    Z = compute_all_Zj(Tk, mean, sigma, x0, xk_plus_1)
    Uk, Xk = compute_uk(Z, mean, sigma, Tk, x0, xk_plus_1, number_of_elements_linespace)
    sk = np.exp(Uk)
    return sk, Xk


def perform_ARS_log_normal(mean, sigma, number_of_samples, Tk, x0, xk_plus_1, plot_every_step=False):
    x_samples = []
    number_of_elements_linespace = 10000
    while len(x_samples) < number_of_samples:
        if plot_every_step:
            X = np.linspace(x0, xk_plus_1, 1000)
            Y = h_log_normal(X, mean, sigma)
            plt.plot(X, Y, label='h(x)')

            Tk_ext = np.array([x0])
            Tk_ext = np.append(Tk_ext, Tk)
            Tk_ext = np.append(Tk_ext, [xk_plus_1])
            Y = h_log_normal(Tk_ext, mean, sigma)
            plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

            Z = compute_all_Zj(Tk, mean, sigma, x0, xk_plus_1)
            Uk, Xk = compute_uk(Z, mean, sigma, Tk, x0, xk_plus_1, number_of_elements_linespace)
            plt.plot(Xk, Uk, label='Lines Uk')
            Lk, Xk = compute_lk(mean, sigma, Tk, x0, xk_plus_1)
            plt.plot(Xk, Lk, label='Lines lk')

            if len(x_samples) > 0:
                Y = []
                for x_samp in x_samples:
                    y = compute_uk_single_x(x_samp, Z, mean, sigma, Tk, x0, xk_plus_1)
                    Y.append(y)
                plt.scatter(x_samples, Y, label='Samples')

            plt.legend()
            plt.show()

        Zk = compute_all_Zj(Tk, mean, sigma, x0, xk_plus_1)

        Sk, Xk = compute_sk_sample_x(Tk, mean, sigma, x0, xk_plus_1, number_of_elements_linespace)
        x_proposal = sample_from_sk(Sk, Xk, plot_every_step, Tk)

        lk_x_proposal = compute_lk_single_x(x_proposal, mean, sigma, Tk, x0, xk_plus_1)
        uk_x_proposal = compute_uk_single_x(x_proposal, Zk, mean, sigma, Tk, x0, xk_plus_1)

        p = np.exp(lk_x_proposal - uk_x_proposal)
        w = np.random.uniform()
        if p > w:
            x_samples.append(x_proposal)
            # Uk, Xk = compute_uk(Zk, mean, sigma, Tk, x0, xk_plus_1, number_of_elements_linespace)
            # plt.plot(Xk, np.exp(Uk), label='Lines Uk')
            # plt.scatter(x_samples, np.exp(h_log_normal(np.array(x_samples), mean, sigma)), label='Samples')
            # plt.legend()
            # plt.show()
        else:
            Tk.append(x_proposal)
            Tk = sorted(Tk)

    X = np.linspace(x0, xk_plus_1, 1000)
    Y = h_log_normal(X, mean, sigma)
    plt.plot(X, Y, label='h(x)')

    Tk_ext = np.array([x0])
    Tk_ext = np.append(Tk_ext, Tk)
    Tk_ext = np.append(Tk_ext, [xk_plus_1])
    Y = h_log_normal(Tk_ext, mean, sigma)
    plt.scatter(Tk_ext, Y, label='Elemnts in Tk')

    Z = compute_all_Zj(Tk, mean, sigma, x0, xk_plus_1)
    Uk, Xk = compute_uk(Z, mean, sigma, Tk, x0, xk_plus_1, number_of_elements_linespace)
    plt.plot(Xk, Uk, label='Lines Uk')
    Lk, Xk = compute_lk(mean, sigma, Tk, x0, xk_plus_1)
    plt.plot(Xk, Lk, label='Lines lk')

    if len(x_samples) > 0:
        Y = []
        for x_samp in x_samples:
            y = compute_uk_single_x(x_samp, Z, mean, sigma, Tk, x0, xk_plus_1)
            Y.append(y)
        plt.scatter(x_samples, Y, label='Samples')

    plt.legend()
    plt.show()

    print "Number of elements in Tk: " + str(len(Tk))

    Y = scipy.stats.norm.pdf(x_samples, loc=mean, scale=sigma)
    plt.scatter(x_samples, Y, label='samples')
    Y = scipy.stats.norm.pdf(Tk, loc=mean, scale=sigma)
    plt.scatter(Tk, Y, label='samples')

    X = np.linspace(-1000, 1000, 10000)
    Y = scipy.stats.norm.pdf(X, loc=mean, scale=sigma)

    plt.plot(X, Y, label='Function')

    plt.legend()
    plt.show()

    return x_samples


def main():
    mean = 100.
    sigma = 10.
    Tk = [-30, 170]
    x0 = -500
    xk_plus_1 = 700
    # X = np.linspace(x0, xk_plus_1, 10000)
    # Y = sc.stats.norm.pdf(X, loc=mean, scale=sigma)
    # plt.plot(X, Y, label='Normal Distribution mean = ' + str(mean) + " variance = " + str(sigma))
    # plt.legend()
    # plt.show()

    x_samples = perform_ARS_log_normal(mean, sigma, 10000, Tk, x0, xk_plus_1, plot_every_step=False)

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

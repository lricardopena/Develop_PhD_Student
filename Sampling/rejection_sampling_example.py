import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats


def f(X):
    realmean = np.array([20, 30, 50])
    realcov = np.array([2, 4, 6])
    realPik = np.array([1. / 2, 1. / 3, 1. / 4])
    realPik = realPik / np.sum(realPik)
    Y = np.zeros_like(X)
    for k in range(len(realmean)):
        Y += realPik[k] * sc.stats.norm.pdf(loc=realmean[k], scale=realcov[k], x=X)

    return Y


def g_u(X):
    Y = 4.8 * sc.stats.norm.pdf(loc=20, scale=20, x=X)
    return Y


def main():
    realmean = np.array([20, 30, 50])
    realcov = np.array([2, 4, 6])
    realPik = np.array([1. / 2, 1. / 3, 1. / 4])
    realPik = realPik / np.sum(realPik)

    # plt.legend()
    # plt.show()

    X_sampled = []
    X_rejected = []
    N = 100
    while len(X_sampled) <= N:
        x_proposal = sc.stats.norm.rvs(loc=20, scale=20, size=1)[0]
        u = sc.random.uniform(size=1)[0]

        if u <= f(x_proposal) / g_u(x_proposal):
            X_sampled.append(x_proposal)
        else:
            X_rejected.append(x_proposal)

    X = np.linspace(10, 120)
    Y = f(X)

    Y_envelop = g_u(X)

    plt.plot(X, Y, label='Real pdf')
    plt.plot(X, Y_envelop, label='Envelop pdf')

    Y = g_u(X_sampled)
    Y_f = f(X_sampled)

    plt.scatter(X_sampled, Y, label='Accepted Samples, projected in Envelop Function', color='Blue')
    plt.scatter(X_sampled, Y_f, label='Accepted Samples, projected in real function', color='Blue')
    plt.legend()
    plt.show()

    X = np.linspace(10, 120)
    Y = f(X)

    Y_envelop = g_u(X)

    plt.plot(X, Y, label='Real pdf')
    plt.plot(X, Y_envelop, label='Envelop pdf')

    Y = g_u(X_rejected)
    Y_f = f(X_rejected)
    plt.scatter(X_rejected, Y, label='Rejected Samples, projected in Envelop Function', color='Red')
    plt.scatter(X_rejected, Y_f, label='Rejected Samples, projected in real function', color='Red')

    print "Number of samples: " + str(len(X_sampled))
    print "Number of rejection points: " + str(len(X_rejected))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

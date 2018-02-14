import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, 1))
    for i in xrange(N):
        randomNumber = np.random.uniform()
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > randomNumber:
                xi = norm.rvs(loc=means[index], scale=cov[index], size=1)
                sampling[i] = xi
                break
            index += 1

    return sampling



def GEM(alpha, lenght):
    pi = []
    betas = []
    for i in range(0, lenght-1):
        betak = beta.rvs(1,alpha)
        pik = 1
        for betal in betas:
            pik *= (1-betal)
        betas.append(betak)
        pik *= betak
        pi.append(pik)
    pi.append(1-np.sum(pi))

    return np.array(pi)

def samplingZj(K, mus, sigmas,W,X,Y,rest):

    np.random.multinomial(M, pi, size=1)

def phiFunction(X, omegas):
    argument = X.reshape((X.shape[0],1)).dot(omegas.T)
    resultCos = np.cos(argument)
    resultSin = np.sin(argument)
    resultPhi = np.column_stack((resultCos, resultSin))

    return 1./np.sqrt(len(omegas))*resultPhi

def f(x):
    phi = phiFunction(x, omegas)
    mean = phi.dot(beta)
    Yi = []
    for m in mean:
        Yi.append(norm.rvs(loc=m, scale=1, size=1)[0])
    return np.array(Yi)

def trueKernel(X):
    return np.exp(-1.0/8 * X**2)*(1.0/2 + 1.0/2 * np.cos(3.0/4*np.pi*X))

means, cov, pik = np.array([0,2.*np.pi/6, 3.0/4* np.pi]), np.array([1.0/4, 1.0/4, 1.0/4]), np.array([1./3,1./3,1./3])
N = 1000
M = 250
sigma = 0.0001
Xi = norm.rvs(loc=0, scale=4, size=N)
omegas = samplingGMM(N=M, means=means,cov=cov, pi=pik)
beta = np.ones(2*M)
Yi = f(Xi)
plt.plot(Xi, Yi, 'b.')
plt.show()

#Plot the true kernel
X = np.arange(-7,7,0.001)
plt.plot(X, trueKernel(X))
plt.show()



pi = GEM(1,6)

Zj = samplingZj(pi, 10)
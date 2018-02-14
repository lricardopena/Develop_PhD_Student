import numpy as np
from scipy.stats import invgamma




def scaledinvchisquare(niu, tausquare):
    return invgamma.rvs(a=niu/2.,scale=niu*tausquare/2., size=1)[0]


def returnBeta_sigma(X, Y, k):
    N = len(X)
    Q, R = np.linalg.qr(X)
    Rinv = np.linalg.inv(R)

    # Vbeta = np.linalg.inv(X.T.dot(X))
    Vbeta = Rinv.dot(Rinv.T)
    # betahat = Vbeta.dot(X.T.dot(Y))
    betahat = Rinv.dot(Q.T.dot(Y))
    sSquare = 1. / (N - k) * (Y - X.dot(betahat)).T.dot(Y - X.dot(betahat))
    sigma = scaledinvchisquare(N - k, sSquare)

    betaProposal = np.random.multivariate_normal(betahat, Vbeta * (sigma ** 2))

    return betaProposal, sigma


d = 3
originalBeta = np.random.uniform(size=d)

originalSigma = 0.000001

N = 10000

X = []
Y = []
for i in range(N):
    x = np.random.uniform(size=d)
    X.append(x)
    Y.append(np.random.normal(x.T.dot(originalBeta), originalSigma, size=1)[0])
X = np.array(X)
Y = np.array(Y)

betaProposal, sigmaProposal = returnBeta_sigma(X,Y,d-1)

print "Beta proposal, sigma: " +  str([betaProposal, sigmaProposal])
print "Real beta: " + str([originalBeta,originalSigma])
print "Error beta: " + str(np.sqrt(np.linalg.norm(betaProposal - originalBeta)))
print "Error sigma: " + str(np.abs((sigmaProposal - originalSigma)))

Y_new = []
for x in X:
    Y_new.append(np.random.normal(x.T.dot(originalBeta), originalSigma, size=1)[0])

print np.allclose(Y, Y_new)

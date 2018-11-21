import logregresion
import numpy as np
import matplotlib.pyplot as plt
def __main():
    size1 = 100*100
    size2 = 100*3
    X1 = np.random.multivariate_normal(np.array([0,0]), np.array([[1,0], [0,1]]), size=size1)
    X2 = np.random.multivariate_normal(np.array([10,10]), np.array([[1,0], [0,1]]), size=size2)
    Y = np.append(np.zeros(len(X1)), np.tile(1, len(X2)))
    X = np.vstack((X1, X2))
    # plt.scatter(X.T[0], X.T[1])
    # plt.show()
    log = logregresion.logreg(X, Y)


    X1 = np.random.multivariate_normal([-10, -10], [[1, 0], [0,1]], size=2000)
    X2 = np.random.multivariate_normal(np.array([100, 100]), np.array([[1, 0], [0, 1]]), size=40)
    Y = np.append(np.zeros(len(X1)), np.tile(1, len(X2)))
    X = np.vstack((X1, X2))
    X = np.column_stack((X, Y))
    np.random.shuffle(X)
    Y = log.predict_Y(X.T[:2].T)
    print (Y)



if __name__ == '__main__':
        __main()
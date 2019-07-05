import matplotlib.pyplot as plt
from Develop_PhD_Student.BaNK.BaNK_classification import bank_classification, bank_locally_classification
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
import numpy as np
import gc
import time


def __do_train_test_locally(X, y, M1, M2, swaps, name_):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    bank_locally = bank_locally_classification(X_train, y_train, M1, M2, bias=0.8)
    print("Start train ---------------------------" + name_ + "-------------------------Time: " + str(time.strftime("%c")) + "----------")
    bank_locally.learn_kernel(swaps, X, y, name_)
    print("End train ---------------------------" + name_ + "-------------------------Time: " + str(time.strftime("%c")) + "----------")
    score = str(bank_locally.score(X_test, y_test))
    print("score: " + name_ + " " + score + " swaps: " + str(swaps) + " M1: " + str(M1) + " M2: " + str(M2))
    print("Confussion matrix test set: ")
    print(bank_locally.return_confussion_matrix(X_test, y_test))
    print("Confussion matrix full set: ")
    print(bank_locally.return_confussion_matrix(X, y))
    bank_locally.save_prediction(X, y, name_ + "_Swaps_" + str(swaps) + "_score_" + score + "_locally.csv")
    gc.collect()
    del bank_locally, X, y, X_train, X_test, y_train, y_test


def __do_train_test(X, y, M, swaps, name_):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    bank = bank_classification(X_train, y_train, M, bias=0.8)
    print("Start train ---------------------------" + name_ + "-------------------------Time: " + str(time.strftime("%c")) + "----------")
    bank.learn_kernel(swaps, X_train, y_train, name_)
    print("Start train ---------------------------" + name_ + "-------------------------Time: " + str(time.strftime("%c")) + "----------")
    score = str(bank.score(X_test, y_test))
    print("score: " + name_ + " " + score + " swaps: " + str(swaps) + " M: " + str(M))
    print("Confussion matrix test set: ")
    print(bank.return_confussion_matrix(X_test, y_test))
    print("Confussion matrix full set: " )
    print(bank.return_confussion_matrix(X, y))
    bank.save_prediction(X, y, name_ + "_Swaps_" + str(swaps) + "_score_" + score + "_stationary.csv")
    gc.collect()

    bank.return_ROC_curve(X_test, y_test, plot=True)
    del bank, X, y, X_train, X_test, y_train, y_test



def cancer_breast():
    X, y = load_breast_cancer(True)
    return X, y


def credit_g():
    X, y = datasets.fetch_openml(data_id=31, return_X_y=True)
    y[np.where(y == 'good')] = 1
    y[np.where(y == 'bad')] = 0

    X, y = np.array(X, float), np.array(y, float)
    return X, y


def blood_transfusion():
    X, y = datasets.fetch_openml(data_id=1464, return_X_y=True)
    X, y = np.array(X, float), np.array(y, float)
    y = y - 1
    return X, y


def kr_vs_kp():
    X, y = datasets.fetch_openml(data_id=3, return_X_y=True)
    y[np.where(y == 'won')] = 1
    y[np.where(y == 'nowin')] = 0

    X, y = np.array(X, float), np.array(y, int)
    return X, y


def phoneme():
    X, y = datasets.fetch_openml(data_id=1489, return_X_y=True)
    X, y = np.array(X, float), np.array(y, float)
    y = y - 1
    return X, y


def eeg_eye_state():
    X, y = datasets.fetch_openml(data_id=1471, return_X_y=True)
    X, y = np.array(X, float), np.array(y, float)
    y = y - 1
    return X, y


def electricity():
    X, y = datasets.fetch_openml(data_id=151, return_X_y=True)
    y[np.where(y == 'UP')] = 1
    y[np.where(y == 'DOWN')] = 0
    y = y.astype(int)
    return X, y


def __xor(M, swaps):
    # XOR
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                         np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(6000, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    bank_cl = bank_classification(X, Y, M)
    bank_cl.learn_kernel(swaps)
    print("Score: " + str(bank_cl.score(X, Y)))
    print("Swaps: " + str(swaps))
    Z1 = bank_cl.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)
    Z1 = Z1.reshape(xx.shape)

    plt.subplot(1, 1, 1)
    image = plt.imshow(Z1, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
    # contours = plt.contour(xx, yy, Z1, levels=[0.5], linewidths=2,              colors=['k'])
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors=(0, 0, 0))
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)

    plt.tight_layout()
    plt.show()
    gc.collect()


def __xor_locally(M1, M2, swaps):
    # XOR
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                         np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(6000, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    bank_cl = bank_locally_classification(X, Y, M1, M2)
    bank_cl.learn_kernel(swaps)
    print("Score: " + str(bank_cl.score(X, Y)))
    print("Swaps: " + str(swaps))
    Z1 = bank_cl.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)
    Z1 = Z1.reshape(xx.shape)

    plt.subplot(1, 1, 1)
    image = plt.imshow(Z1, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
    # contours = plt.contour(xx, yy, Z1, levels=[0.5], linewidths=2,              colors=['k'])
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors=(0, 0, 0))
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    gc.collect()
    plt.tight_layout()
    plt.show()



def linearly_moons_circles(M, swaps):
    h = .05  # step size in the mesh
    N = 4000
    X, y = make_classification(n_samples=N, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(n_samples=N, noise=0.3, random_state=0),
                make_circles(n_samples=N, noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]
    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), 2, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers

        ax = plt.subplot(len(datasets), 2, i)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        bank_cl = bank_classification(X_train, y_train, M)
        bank_cl.learn_kernel(swaps)
        bank_cl.predict_new_X(X_test, real_Y=y_test)
        score = bank_cl.score(X_test, y_test)
        Z1 = bank_cl.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z1 = Z1.reshape(xx.shape)
        ax.contourf(xx, yy, Z1, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title("BaNK")
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.show()
    gc.collect()


def common_examples_locally(M1, M2, swaps):
    print("Locally stationary kernel Learning: \n")
    #__xor_locally(M1, M2, swaps)

    # X, y = electricity()
    # __do_train_test_locally(X, y, M1, M2, swaps, "electricity")
    # del X, y

    # X, y = eeg_eye_state()
    # __do_train_test_locally(X, y, M1, M2, swaps, "egg-eye-state")
    # del X, y

    # X, y = kr_vs_kp()
    # __do_train_test_locally(X, y, M1, M2, swaps, "KR vs KP")
    # del X, y

    X, y = cancer_breast()
    __do_train_test_locally(X, y, M1, M2, swaps, "Cancer Breast")
    del X, y

    X, y = credit_g()
    __do_train_test_locally(X, y, M1, M2, swaps, "Credit-g")
    del X, y

    X, y = blood_transfusion()
    __do_train_test_locally(X, y, M1, M2, swaps, "blood transfusion")
    del X, y



    print("End Locally stationary kernel Learning: \n")


def common_examples(M, swaps):
    print("Stationary kernel Learning: \n")
    #__xor(M, swaps)
    # linearly_moons_circles(M, swaps)
    # X, y = cancer_breast()
    # __do_train_test(X, y, M, swaps, "cancer breast")
    # del X, y
    #
    # X, y = credit_g()
    # __do_train_test(X, y, M, swaps, "credit-g")
    # del X, y
    #
    # X, y = blood_transfusion()
    # __do_train_test(X, y, M, swaps, "blood transfusion")
    # del X, y

    X, y = electricity()
    __do_train_test(X, y, M, swaps, "electricity")
    del X, y

    # X, y = eeg_eye_state()
    # __do_train_test(X, y, M, swaps, "egg-eye-state")
    # del X, y

    X, y = kr_vs_kp()
    __do_train_test(X, y, M, swaps, "kr vs kp", )
    del X, y

    print("End Stationary kernel Learning: \n")

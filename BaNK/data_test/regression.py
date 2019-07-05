from Develop_PhD_Student.BaNK import BaNK_regression
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import gc


def __do_train_test(X, y, M, swaps, name_):
    print("Start train ---------------------------" + name_ + "-------------------------Time: " + str(
        time.strftime("%c")) + "----------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    bank = BaNK_regression.bank_regression(X_train, y_train, M) # Bank_Locally_Regression
    bank.learn_kernel(swaps, X, y, name_)

    MSE_train = bank.return_mse(X_train, y_train)
    print("MSE train: " + name_ + " " + str(MSE_train) + " swaps: " + str(swaps) + " M: " + str(M))

    MSE_test = bank.return_mse(X_test, y_test)
    print("MSE test: " + name_ + " " + str(MSE_test) + " swaps: " + str(swaps) + " M: " + str(M))

    bank.save_prediction(X, y, name_ + "_Swaps_" + str(swaps) + "_score_" + MSE_test + "_stationary.csv")
    del X, y, bank
    gc.collect()


def __do_train_test_locally(X, y, M1, M2, swaps, name_):
    #Falta implementar bien el BaNK locally así como el BaNK classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    bank = BaNK_regression.Bank_Locally_Regression(X_train, y_train, M1, M2) # Bank_Locally_Regression
    bank.learn_kernel(swaps, X, y, name_)
    y_pred = bank.predict(X_test)
    y_min, y_max = min(y), max(y)
    print("MSE: " + name_ + " " + str(mean_squared_error(y_test, y_pred)))
    del X, y, X_train, X_test, y_train, y_test
    print("y_min: " + str(y_min) + " y_max: " + str(y_max) + " Range:" + str(y_max - y_min))


def load_mauna_loa_atmospheric_co2():
    ml_data = datasets.fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    del ml_data, m, month_float, ppmv_sums, ppmvs, y, counts
    return months, avg_ppmvs


def mauna_atmospheric(M, swaps):
    X, y = load_mauna_loa_atmospheric_co2()
    X = X.reshape(len(X), )
    bank = BaNK_regression.bank_regression(X, y, M)
    bank.learn_kernel(swaps, X, y, "Mauna LOA CO2")

    X_ = np.linspace(X.min(), X.max() + 3, 1000)[:, np.newaxis]
    y_pred = bank.predict(X_)
    bank.sample_beta_sigma()
    y_std = bank.get_beta_sigma()[1]

    # Illustration
    plt.scatter(X, y, c='k')
    plt.plot(X_, y_pred)
    plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                     alpha=0.5, color='k')
    plt.xlim(X_.min(), X_.max())
    plt.xlabel("Year")
    plt.ylabel(r"CO$_2$ in ppm")
    plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
    plt.tight_layout()
    plt.legend()
    plt.show()


def common_examples(M, swaps):
    X, y = load_mauna_loa_atmospheric_co2()
    X = X.reshape(len(X), )
    name_ = "Mauna LOA CO2"
    print("Start train --------------------------- Mauna LOA CO2 -------------------------Time: " + str(
        time.strftime("%c")) + "----------")
    bank = BaNK_regression.bank_regression(X, y, M)  # Bank_Locally_Regression
    bank.learn_kernel(swaps, X, y, name_)

    MSE_train = bank.return_mse(X, y)

    print("MSE train: " + name_ + " " + str(MSE_train) + " swaps: " + str(swaps) + " M: " + str(M))

    X_ = np.linspace(X.min(), X.max() + 3, 1000)[:, np.newaxis]
    del X, y
    bank.save_prediction_noTrue(X_, name_ + "_Swaps_" + str(swaps) + "_MSE_" + str(MSE_train) + "_stationary.csv")
    del bank
    # bank = null
    gc.collect()

    dataset = datasets.fetch_california_housing()
    X_full, y_full = dataset.data, dataset.target
    del dataset
    __do_train_test(X_full, y_full, M, swaps, "california_houses")

    X, y = datasets.load_boston(True)
    __do_train_test(X, y, M, swaps, "Boston house-price")
    del X, y

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    __do_train_test(X, y, M, swaps, "Diabetes")
    del X, y

def common_examples_locally(M1, M2, swaps):
    X, y = load_mauna_loa_atmospheric_co2()
    X = X.reshape(len(X), )
    __do_train_test_locally(X, y, M1, M2, swaps, "Mauna LOA CO2")
    del X, y

    dataset = datasets.fetch_california_housing()
    X_full, y_full = dataset.data, dataset.target
    del dataset
    __do_train_test_locally(X_full, y_full, M1, M2, swaps, "california_houses")
    del X, y

    X, y = datasets.load_boston(True)
    __do_train_test_locally(X, y, M1, M2, swaps, "Boston house-price")
    del X, y

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    __do_train_test_locally(X, y, M1, M2, swaps, "Diabetes")
    del X, y

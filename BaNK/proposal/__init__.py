import BaNK_locally as bn
import numpy as np

def __main():
    X = np.random.rand(1000, 2)
    Y = (np.random.rand(1000)) *100
    bank = bn.bank(X, Y, 250, 100)

    print("stop")


if __name__ == '__main__':
    __main()
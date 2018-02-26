import matplotlib.pyplot as plt
import numpy as np

X1 = np.random.uniform(-200, -100, 50)
X2 = np.random.uniform(100, 200, 50)

X = range(-220, 220)
plt.plot(X, np.zeros_like(X))

plt.scatter(X1, np.zeros_like(X1), label='Clase 1')
plt.scatter(X2, np.zeros_like(X2), label='Clase 2')

plt.scatter([0], [0], label='Hiperplano')
plt.legend()
plt.show()

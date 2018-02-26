import matplotlib.pyplot as plt
import numpy as np

sample_size = 500

X1 = np.random.multivariate_normal(mean=[20, 20], cov=[[10, 0], [0, 3]], size=sample_size)
X2 = np.random.multivariate_normal(mean=[-20, -10], cov=[[3, 0], [0, 10]], size=sample_size)

Y1 = np.ones(sample_size)
Y2 = -np.ones(sample_size)

Y = np.append(Y1, Y2)

X = np.append(X1, X2, axis=0)
X = np.column_stack([np.ones(sample_size * 2), X])

plt.scatter(X1.T[0], X1.T[1], label='Clase 1')
plt.scatter(X2.T[0], X2.T[1], label='Clase 2')

# plt.scatter([0], [0], label='Hiperplano')
w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

x1 = np.linspace(-40, 40)
x2 = (-w[1] * x1 - w[0]) / w[2]

plt.plot(x1, x2, label='Hiperplano de desicion')

plt.legend()
plt.show()

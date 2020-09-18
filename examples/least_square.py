import numpy as np
from numpy import asarray
import random
import matplotlib.pyplot as plt

x = 3

a = np.arange(-2, 2, .25)
a = a.reshape(-1, 1)

b = a * x + 1. * np.random.randn(*a.shape)

plt.plot(a, a * x)
plt.plot(a, b)

U, S, VT = np.linalg.svd(a, full_matrices=False)

x_tilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(a, a * x_tilde)

plt.show()

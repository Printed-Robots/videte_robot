import numpy as np

X = np.random.rand(5, 3)
U, S, V = np.linalg.svd(X, full_matrices=True)
Uhat, Shat, Vhat = np.linalg.svd(X, full_matrices=False)

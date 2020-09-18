import numpy as np
from numpy import linalg as LA
# inverted pendulum sample
for d in np.arange(0, 1, 0.1):
    print("d: " + str(d))
    Ad = np.array([[0, 1], [-1, -d]])

    Au = np.array([[0, 1], [1, -d]])

    w, v = LA.eig(Ad)
    print("Ad:" + str(w))

    w, v = LA.eig(Au)
    print("Au:" + str(w))

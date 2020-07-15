import control
import numpy as np
from numpy import linalg as LA

B = np.array([0, 1])
B = B.reshape(-1, 1)

Ac = np.array([[1, 1], [-1, 0]])

Anc = np.array([[1, 0], [0, 1]])

print("controllable: ")
print(LA.matrix_rank(control.ctrb(Ac, B)))
print("not controllable: ")
print(LA.matrix_rank(control.ctrb(Anc, B)))

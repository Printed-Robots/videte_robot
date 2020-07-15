#!/usr/bin/env python
import numpy as np
from numpy import linalg as LA
import control

B = np.array([0, 1])
B = B.reshape(-1, 1)

Ac = np.array([[1, 1], [0, 2]])

Anc = np.array([[1, 0], [0, 2]])

print("controllable: ")
Cc = control.ctrb(Ac, B)
print(LA.matrix_rank(Cc))
print("not controllable: ")
Cnc = control.ctrb(Anc, B)
print(LA.matrix_rank(Cnc))

import numpy as np

n = 7  # x
m = 10

p = 5  # y
q = 2  # u

x = np.zeros((n, 1))  # internal state
x_dot = np.zeros((n, 1))  # next internal state
y = np.zeros((p, 1))  # measurement update
u = np.zeros((q, 1))  # input data

A = np.zeros((n, m))  # System
B = np.zeros((n, q))  # Controller characteristics
C = np.zeros((p, n))  # Measurement to state

K = np.zeros((q, n))  # Controller

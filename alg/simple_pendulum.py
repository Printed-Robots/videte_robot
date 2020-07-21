import math
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import sympy

# TODO: Consider to do a dimensional analysis https://docs.sympy.org/latest/modules/physics/units/examples.html#dimensional-analysis

sympy.init_printing()

t = sympy.symbols('t')
x = sympy.Function('x')(t)
y = sympy.Function('y')(t)
θ = sympy.Function('θ')(t)

# Constants
l = sympy.symbols('L')
g = sympy.symbols('g')

A = sympy.Matrix([[0, 1], [-(g/l) * sympy.cos(θ), 0]])

linearize = [(sympy.sin(θ), 0), (sympy.cos(θ), 1), (θ.diff(t), 0),
             (θ.diff(t)**2, 0), (θ.diff(t, t), 0), (y.diff(t), 0), (y.diff(t, t), 0)]

A_lin = sympy.simplify(A.subs(linearize))

constants = {
    g: 9.81,
    l: 1.0
}

Jacobian = A_lin.subs(constants)

x_0 = sympy.Matrix([0.0, 0.1])

dt = 0.01

timeline = np.arange(0.0, 5, dt)
result = sympy.zeros(len(x_0), len(timeline))
resultA = sympy.zeros(len(x_0), len(timeline))

result[:, 0] = x_0
resultA[:, 0] = x_0

for i in range(len(timeline) - 1):
    linearize = [(θ, resultA[0, i])]

    result[:, i + 1] = Jacobian * result[:, i] * dt + result[:, i]
    resultA[:, i + 1] = A.subs(linearize).subs(constants) * \
        resultA[:, i] * dt + resultA[:, i]

# X_DOT = Jacobian * X

fig, plots = plt.subplots(2)

for row in range(len(x_0)):
    plots[0].plot(timeline, result.row(row).T, label='J (' + str(row) + ')')
    plots[1].plot(timeline, resultA.row(row).T, label='A (' + str(row) + ')')

plots[0].axis([-0.01, 5, -1, 2])
plots[1].axis([-0.01, 5, -1, 2])
plots[0].legend()
plots[1].legend()

plt.show()

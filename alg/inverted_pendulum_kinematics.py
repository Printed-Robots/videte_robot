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

F = sympy.Function('F')(t)

# Masses
m_cart, m_m = sympy.symbols('m_cart m_m')

# Dimensions
l = sympy.symbols('L')

# Constants
g = sympy.symbols('g')


# Y positions
y_cart = y
y_m = y_cart + l * sympy.cos(θ)

# X positions
x_cart = x
x_m = x_cart + l * sympy.sin(θ)

####################
# Potential Energy #
####################

# Potential Energy of Videte Wheel Concept
V = m_m * g * y_m + m_cart * g * y_cart

##################
# KINETIC ENERGY #
##################

# Translational kinetic energy
W_cart = (1/2) * m_cart * (sympy.diff(x_cart, t)**2 + sympy.diff(y_cart, t)**2)
W_m = (1/2) * m_m * (sympy.diff(x_m, t)**2 + sympy.diff(y_m, t)**2)

# Kinetic Energy
T = W_cart + W_m

############
# Lagrange #
############

L = T - V

x_dot = sympy.diff(x, t)
θ_dot = sympy.diff(θ, t)
θ_2dot = sympy.diff(θ, t, t)

# Equations of Motion
eq_y = sympy.diff(sympy.diff(L, sympy.diff(y, t)), t) - sympy.diff(L, y)
eq_x = sympy.diff(sympy.diff(L, sympy.diff(x, t)), t) - sympy.diff(L, x) - F
eq_x_dot = sympy.diff(sympy.diff(L, sympy.diff(x_dot, t)),
                      t) - sympy.diff(L, x_dot)
eq_θ = sympy.diff(sympy.diff(L, sympy.diff(θ, t)), t) - sympy.diff(L, θ)
eq_θ_dot = sympy.diff(sympy.diff(L, sympy.diff(θ_dot, t)),
                      t) - sympy.diff(L, θ_dot)
eq_θ_2dot = sympy.diff(sympy.diff(L, sympy.diff(θ_2dot, t)),
                       t) - sympy.diff(L, θ_2dot)

eqs = [eq_y, eq_x, eq_x_dot, eq_θ, eq_θ_dot, eq_θ_2dot]
symbols = [y, x, x_dot, θ, θ_dot, θ_2dot]


def createMatrix(eqs: list, symbols: list) -> sympy.Matrix:
    A = sympy.zeros(len(eqs), len(eqs))
    for i, symbol in enumerate(symbols, start=0):
        for j, eq in enumerate(eqs, start=0):
            A[i, j] = sympy.diff(eq, symbol)
    return A


A = createMatrix(eqs, symbols)

linearize = [(sympy.sin(θ), 0), (sympy.cos(θ), 1), (θ.diff(t), 0),
             (θ.diff(t)**2, 0), (θ.diff(t, t), 0), (y.diff(t), 0), (y.diff(t, t), 0)]

A_lin = sympy.simplify(A.subs(linearize))

constants = {
    g: 9.81,
    l: 1.0,
    m_m: 2.0,
    m_cart: 2.0
}

Jacobian = A_lin.subs(constants)

x_0 = sympy.Matrix([1.0, 0.5, 0.3, -0.02, 2.15, 0])

dt = 0.01

timeline = np.arange(0.0, 5, dt)
result = sympy.zeros(len(x_0), len(timeline))
resultA = sympy.zeros(len(x_0), len(timeline))

result[:, 0] = x_0
resultA[:, 0] = x_0

for i in range(len(timeline) - 1):
    linearize = [(θ.diff(t, t), resultA[5, i]), (θ.diff(t), resultA[4, i]),
                 (θ, resultA[3, i]), (x.diff(t, t), 0), (x.diff(t), resultA[2, i]), (x, resultA[1, i]), (y.diff(t, t), 0), (y.diff(t), 0)]

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

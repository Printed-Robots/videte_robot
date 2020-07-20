import math
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import sympy

# TODO: Consider to do a dimensional analysis https://docs.sympy.org/latest/modules/physics/units/examples.html#dimensional-analysis

sympy.init_printing()

t = sympy.symbols('t')
# x = sympy.Function('x')(t)
y = sympy.Function('y')(t)
θ = sympy.Function('θ')(t)
ρ = sympy.Function('ρ')(t)
ω_iw = sympy.Function('ω_iw')(t)

# Masses
m_ow, m_ow_rot, m_iw, m_iw_rot, m_m = sympy.symbols(
    'm_ow m_owr m_iw m_iwr m_m')

# Dimensions
l_iw_m, r_1, r_2, r_3 = sympy.symbols('l_iwm r_1 r_2 r_3')

l_ow_iw = r_2 - r_1

# Constants
g = sympy.symbols('g')

i = r_2 / r_1

# Linking rotation of outer and inner wheel
ω_ow = (1/i) * ω_iw

# Y positions
y_ow = r_3 + y
y_iw = y_ow - l_ow_iw * sympy.cos(ρ)
y_m = y_iw + l_iw_m * sympy.cos(θ)

x = sympy.integrate(r_3 * (ω_ow * sympy.diff(ρ, t)), (t, 0, t))

# X positions
x_ow = x
x_iw = x_ow + l_ow_iw * sympy.sin(ρ)
x_m = x_iw + l_iw_m * sympy.sin(θ)

####################
# Potential Energy #
####################

# Potential Energy of Videte Wheel Concept
V = m_m * g * y_m + m_ow * g * y_ow + m_iw * g * y_iw

##################
# KINETIC ENERGY #
##################

# Mass dampening inertia
J_ow = (1/2) * m_ow_rot * (r_2**2 + r_3**2)
J_iw = (1/2) * m_iw_rot * r_1**2

# Kinetic energy of rotating cylinder
W_owr = (1/2) * J_ow * ω_ow**2
W_iwr = (1/2) * J_iw * ω_iw**2


# Translational kinetic energy
W_ow = (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2)
W_iw = (1/2) * (sympy.diff(x_iw, t)**2 + sympy.diff(y_iw, t)**2)
W_m = (1/2) * (sympy.diff(x_m, t)**2 + sympy.diff(y_m, t)**2)

# Dampening energy (Heat dissipation)
d_ow, d_iw, d_m, d_owr, d_iwr = sympy.symbols(
    'd_ow d_iw d_m d_owr d_iwr')

W_d_ow = d_ow * (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2)
W_d_iw = d_iw * (1/2) * sympy.diff(ρ, t)**2
W_d_m = d_m * (1/2) * sympy.diff(θ, t)**2
W_d_owr = d_owr * (1/2) * ω_ow**2
W_d_iwr = d_iwr * (1/2) * ω_iw**2

W_heat = W_d_ow + W_d_iw + W_d_m + W_d_owr + W_d_iwr
W_heat = 0

# Kinetic Energy
T = W_owr + W_iwr + W_ow + W_iw + W_m - W_heat

############
# Lagrange #
############

L = T - V

θ_dot = sympy.diff(θ, t)
ρ_dot = sympy.diff(ρ, t)
ω_iw_dot = sympy.diff(ω_iw, t)

# Equations of Motion
# eq_x = sympy.diff(sympy.diff(L, sympy.diff(x)), t) - sympy.diff(L, x)
eq_y = sympy.diff(sympy.diff(L, sympy.diff(y, t)), t) - sympy.diff(L, y)
eq_θ = sympy.diff(sympy.diff(L, sympy.diff(θ, t)), t) - sympy.diff(L, θ)
eq_ρ = sympy.diff(sympy.diff(L, sympy.diff(ρ, t)), t) - sympy.diff(L, ρ)
eq_ω_iw = sympy.diff(sympy.diff(L, sympy.diff(ω_iw, t)),
                     t) - sympy.diff(L, ω_iw)
eq_θ_dot = sympy.diff(sympy.diff(L, sympy.diff(θ_dot, t)),
                      t) - sympy.diff(L, θ_dot)
eq_ρ_dot = sympy.diff(sympy.diff(L, sympy.diff(ρ_dot, t)),
                      t) - sympy.diff(L, ρ_dot)
eq_ω_iw_dot = sympy.diff(sympy.diff(
    L, sympy.diff(ω_iw_dot)), t) - sympy.diff(L, ω_iw_dot)

eqs = [eq_y, eq_θ, eq_θ_dot, eq_ρ, eq_ρ_dot, eq_ω_iw, eq_ω_iw_dot]
symbols = [y, θ, θ_dot, ρ, ρ_dot, ω_iw, ω_iw_dot]


def createMatrix(eqs: list, symbols: list) -> sympy.Matrix:
    A = sympy.zeros(len(eqs), len(eqs))
    for i, symbol in enumerate(symbols, start=0):
        for j, eq in enumerate(eqs, start=0):
            A[i, j] = sympy.diff(eq, symbol)
    return A


A = createMatrix(eqs, symbols)

linearize = [
    (sympy.sin(ρ), 0),
    (sympy.cos(ρ), 1),
    (sympy.sin(θ), 0),
    (sympy.cos(θ), 1),
    (θ.diff(t)**2, 0),
    (ρ.diff(t), 0),
    (ρ.diff(t)**2, 0),
    (ρ.diff(t, t), 0),
    (θ.diff(t), 0),
    (θ.diff(t, t), 0),
    (y.diff(t), 0),
    (y.diff(t, t), 0),
    (ω_iw, 0)
]

A_lin = sympy.simplify(A.subs(linearize))

masses = {
    g: 9.81,
    m_ow: 1.0,
    m_ow_rot: 0.5,
    m_iw: 0.3,
    m_iw_rot: 0.1,
    m_m: 2.0
}

lengths = {
    l_iw_m: 2.0,
    r_1: 0.25,
    r_2: 1.0,
    r_3: 1.2
}

dampening = {
    d_ow: 0.01,
    d_iw: 0.01,
    d_m: 0.01,
    d_owr: 0.01,
    d_iwr: 0.01
}

Jacobian = A_lin.subs(masses).subs(lengths).subs(dampening)

x_0 = sympy.Matrix([0.0, 0.1, 0.01, 0.1, 0.001, -1.0, 0.2])

t = np.arange(0.0, 5, 0.01)
result = sympy.zeros(len(x_0), len(t))

result[:, 0] = x_0

for i in range(len(t) - 1):
    result[:, i + 1] = Jacobian * result[:, i] * -0.01 + result[:, i]

# X_DOT = Jacobian * X

for row in range(len(x_0)):
    plt.plot(t, result.row(row).T)

plt.axis([-0.01, 0.1, -0.1, 0.6])
plt.show()

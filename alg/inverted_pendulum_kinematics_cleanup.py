import math
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import sympy


sympy.init_printing()

t = sympy.symbols('t')

# x = sympy.Function('x')(t) defined by ω and ρ
y = sympy.Function('y')(t)
θ = sympy.Function('θ')(t)
ρ = sympy.Function('ρ')(t)
φ = sympy.Function('φ')(t)


# Masses
m_ow, m_ow_rot, m_iw, m_iw_rot, m_m = sympy.symbols(
    'm_ow m_owr m_iw m_iwr m_m')

# Dimensions
l_iw_m, r_1, r_2, r_3 = sympy.symbols('l_iwm r_1 r_2 r_3')
l_ow_iw = r_2 - r_1

# Constants
g = sympy.symbols('g')

# Gear ratio
i = r_2 / r_1

# Linking rotation of outer and inner wheel
φ_iw = φ
φ_ow = (1/i) * φ_iw

ω = φ.diff(t)
ω_iw = ω
ω_ow = (1/i) * ω_iw


# Y positions
y_ow = r_3 + y
y_iw = y_ow - l_ow_iw * sympy.cos(ρ)
y_m = y_iw + l_iw_m * sympy.cos(θ)


# Rotation based x movement
# TODO: Check to be correct
x = r_3 * (φ_ow - ρ)


# X positions
x_ow = x
x_iw = x_ow + l_ow_iw * sympy.sin(ρ)
x_m = x_iw + l_iw_m * sympy.sin(θ)


####################
# Potential Energy #
####################

# Potential Energy of Videte Wheel Concept
V = (m_m * y_m + m_ow * y_ow + m_iw * y_iw) * g


##################
# KINETIC ENERGY #
##################

# Rotating mass inertia
J_ow = (1/2) * m_ow_rot * (r_2**2 + r_3**2)
J_iw = (1/2) * m_iw_rot * r_1**2

# Kinetic energy of rotating cylinder
W_owr = (1/2) * J_ow * ω_ow**2
W_iwr = (1/2) * J_iw * ω_iw**2

# Translational kinetic energy
W_ow = (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2)
W_iw = (1/2) * (sympy.diff(x_iw, t)**2 + sympy.diff(y_iw, t)**2)
W_m = (1/2) * (sympy.diff(x_m, t)**2 + sympy.diff(y_m, t)**2)

# Dampening (e.g. heat dissipation)
d_ow, d_iw, d_m, d_owr, d_iwr = sympy.symbols(
    'd_ow d_iw d_m d_owr d_iwr')

W_d_ow = d_ow * (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2)
W_d_iw = d_iw * (1/2) * sympy.diff(ρ, t)**2
W_d_m = d_m * (1/2) * sympy.diff(θ, t)**2
W_d_owr = d_owr * (1/2) * ω_ow**2
W_d_iwr = d_iwr * (1/2) * ω_iw**2

W_heat = W_d_ow + W_d_iw + W_d_m + W_d_owr + W_d_iwr
# Currently ignored
# W_heat = 0


# Kinetic Energy
T = W_owr + W_iwr + W_ow + W_iw + W_m - W_heat


###############
# State Space #
###############

states = [y, y.diff(t), θ, θ.diff(t), ρ, ρ.diff(t), φ, φ.diff(t)]

############
# Lagrange #
############

L = T - V

f_y = y.diff(t)
L_y = sympy.diff(sympy.diff(L, sympy.diff(y, t)), t) - sympy.diff(L, y)
f_dy = sympy.solve(L_y, y.diff(t, t))[0]

f_θ = θ.diff(t)
L_θ = sympy.diff(sympy.diff(L, sympy.diff(θ, t)), t) - sympy.diff(L, θ)
f_dθ = sympy.solve(L_θ, θ.diff(t, t))[0]

f_ρ = ρ.diff(t)
L_ρ = sympy.diff(sympy.diff(L, sympy.diff(ρ, t)), t) - sympy.diff(L, ρ)
f_dρ = sympy.solve(L_ρ, ρ.diff(t, t))[0]

f_φ = φ.diff(t)
L_φ = sympy.diff(sympy.diff(L, sympy.diff(φ, t)), t) - sympy.diff(L, φ)
f_dφ = sympy.solve(L_φ, φ.diff(t, t))[0]

eqs = [f_y, f_dy, f_θ, f_dθ, f_ρ, f_dρ, f_φ, f_dφ]


def createMatrix(eqs: list, states: list) -> sympy.Matrix:
    A = sympy.zeros(len(eqs), len(eqs))
    for i, eq in enumerate(eqs, start=0):
        for j, state in enumerate(states, start=0):
            A[i, j] = sympy.diff(eq, state)
    return A


# Create A Matrix
A = createMatrix(eqs, states)


# Linearice
linearice = [
    (sympy.sin(ρ), 0),
    (sympy.cos(ρ), 1),
    (sympy.sin(θ), 0),
    (sympy.cos(θ), 1),
    (ρ.diff(t, t), 0),
    (ρ.diff(t)**2, 0),
    (ρ.diff(t), 0),
    (θ.diff(t)**2, 0),
    (θ.diff(t, t), 0),
    (θ.diff(t), 0),
    (y.diff(t, t), 0),
    (y.diff(t), 0)
]

A_lin = sympy.simplify(A.subs(linearice))

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

# Simulation
x_0 = sympy.Matrix([1.0, 0.0, 0.15, 0.2, 0.0, -0.1, 0.2, 0.0])


dt = 0.1

timeline = np.arange(0.0, 5, dt)
result = sympy.zeros(len(x_0), len(timeline))
resultA = sympy.zeros(len(x_0), len(timeline))

result[:, 0] = x_0
resultA[:, 0] = x_0
result[:, 1] = x_0
resultA[:, 1] = x_0

# states = [y, y.diff(t), θ, θ.diff(t), ρ, ρ.diff(t), φ, φ.diff(t)]

for i in range(len(timeline) - 2):
    pre = i
    act = i + 1
    nex = i + 2
    linearice = [
        (y.diff(t, t), resultA[1, act] - resultA[1, pre]),
        (y.diff(t), resultA[1, act]),
        (θ.diff(t, t), resultA[3, act] - resultA[3, pre]),
        (θ.diff(t), resultA[3, act]),
        (θ, resultA[2, act]),
        (ρ.diff(t, t), resultA[5, act] - resultA[5, pre]),
        (ρ.diff(t), resultA[5, act]),
        (ρ, resultA[4, act]),
        (φ.diff(t, t), resultA[7, act] - resultA[7, pre]),
        (φ.diff(t), resultA[7, act]),
        (φ, resultA[6, act])
    ]

    result[:, nex] = Jacobian * result[:, i] * dt + result[:, i]
    resultA[:, nex] = A.subs(linearice).subs(masses).subs(lengths).subs(
        dampening) * resultA[:, act] * dt + resultA[:, act]


fig, plots = plt.subplots(2)

for row in range(len(x_0)):
    plots[0].plot(timeline, result.row(row).T, label='J (' + str(row) + ')')
    plots[1].plot(timeline, resultA.row(row).T, label='A (' + str(row) + ')')

# plots[0].axis([-0.01, 5, -1, 2])
# plots[1].axis([-0.01, 5, -1, 2])
plots[0].legend()
plots[1].legend()

plt.show()

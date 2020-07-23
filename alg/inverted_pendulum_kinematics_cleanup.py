import tqdm
import math
import scipy.integrate as integrate
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
i_gear = r_2 / r_1

# Linking rotation of outer and inner wheel
φ_iw = φ
φ_ow = (1/i_gear) * φ_iw

ω = φ.diff(t)
ω_iw = ω
ω_ow = (1/i_gear) * ω_iw


# Y positions
y_ow = r_3 + y
y_iw = y_ow - l_ow_iw * sympy.cos(ρ)
y_m = y_iw - l_iw_m * sympy.cos(θ)


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

W_d_ow = sympy.integrate(
    d_ow * (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2), t)
W_d_iw = sympy.integrate(d_iw * (1/2) * sympy.diff(ρ, t)**2, t)
W_d_m = sympy.integrate((d_m * (1/2) * sympy.diff(θ, t)**2), t)
W_d_owr = sympy.integrate(d_owr * (1/2) * ω_ow**2, t)
W_d_iwr = sympy.integrate(d_iwr * (1/2) * ω_iw**2, t)

W_heat = W_d_ow + W_d_iw + W_d_m + W_d_owr + W_d_iwr


# Kinetic Energy
T = W_owr + W_iwr + W_ow + W_iw + W_m + W_heat


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
    d_ow: 1.0,
    d_iw: 1.0,
    d_m: 1.0,
    d_owr: 1.0,
    d_iwr: 1.0
}

Jacobian = np.float64(A_lin.subs(masses).subs(lengths).subs(dampening))

# Simulation
x_0 = np.float64([1.0, 0.0, 3.16, 1.0, 0.14, 0.23, 0.0, 0.1])


dt = 0.01

timeline = np.arange(0.0, 10.0, dt)


def applyJ(y, t):
    return Jacobian.dot(y)


solution = integrate.odeint(applyJ, x_0, timeline)

# x_ow, y_ow, x_iw, y_iw, x_m, y_m
positions = sympy.zeros(6, len(timeline))

for tp in tqdm.tqdm(range(len(timeline))):
    results = {
        y: solution[tp, 0],
        φ: solution[tp, 6],
        ρ: solution[tp, 4],
        θ: solution[tp, 2]
    }
    positions[0, tp] = x_ow.subs(lengths).subs(results)
    positions[1, tp] = y_ow.subs(lengths).subs(results)
    positions[2, tp] = x_iw.subs(lengths).subs(results)
    positions[3, tp] = y_iw.subs(lengths).subs(results)
    positions[4, tp] = x_m.subs(lengths).subs(results)
    positions[5, tp] = y_m.subs(lengths).subs(results)

fig, axs = plt.subplots(2)

for row, state in enumerate(states):
    axs[0].plot(timeline, solution[:, row], label=str(state))

axs[1].plot(timeline, positions[0, :].T, label='x_ow')
# axs[1].plot(timeline, positions[1, :].T, label='y_ow')
axs[1].plot(timeline, positions[2, :].T, label='x_iw')
axs[1].plot(timeline, positions[3, :].T, label='y_iw')
axs[1].plot(timeline, positions[4, :].T, label='x_m')
axs[1].plot(timeline, positions[5, :].T, label='y_m')

axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()
plt.xlabel('t')
# plt.axis([-0.01, 5, -0.75, 1.5])
plt.show()

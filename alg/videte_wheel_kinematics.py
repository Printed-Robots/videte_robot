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
l_ow_iw, l_iw_m, r_1, r_2, r_3 = sympy.symbols('l_owiw l_iwm r_1 r_2 r_3')

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

# Equations of Motion
# eq_x = sympy.diff(sympy.diff(L, sympy.diff(x)), t) - sympy.diff(L, x)
eq_y = sympy.diff(sympy.diff(L, sympy.diff(y)), t) - sympy.diff(L, y)
eq_θ = sympy.diff(sympy.diff(L, sympy.diff(θ)), t) - sympy.diff(L, θ)
eq_ρ = sympy.diff(sympy.diff(L, sympy.diff(ρ)), t) - sympy.diff(L, ρ)
eq_ω_iw = sympy.diff(sympy.diff(L, sympy.diff(ω_iw)), t) - sympy.diff(L, ω_iw)

A = sympy.Matrix([
    [sympy.diff(eq_y, y), sympy.diff(eq_ω_iw, y),
     sympy.diff(eq_θ, y), sympy.diff(eq_ω_iw, y)],
    [sympy.diff(eq_y, θ), sympy.diff(eq_ω_iw, θ),
     sympy.diff(eq_θ, θ), sympy.diff(eq_ω_iw, θ)],
    [sympy.diff(eq_y, ρ), sympy.diff(eq_ω_iw, ρ),
     sympy.diff(eq_θ, ρ), sympy.diff(eq_ω_iw, ρ)],
    [sympy.diff(eq_y, ω_iw), sympy.diff(eq_ω_iw, ω_iw),
     sympy.diff(eq_θ, ω_iw), sympy.diff(eq_ω_iw, ω_iw)]
])

linearize = [(sympy.sin(ρ), 0), (sympy.cos(ρ), 1), (sympy.sin(θ), 0), (sympy.cos(θ), 1), (ρ.diff(
    t)**2, 0), (θ.diff(t)**2, 0), (ρ.diff(t, t), 0), (θ.diff(t, t), 0), (y.diff(t, t), 0), (ω_iw, 0)]

sympy.simplify(A.subs(linearize))

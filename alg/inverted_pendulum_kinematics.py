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
W_cart = (1/2) * (sympy.diff(x_cart, t)**2 + sympy.diff(y_cart, t)**2)
W_m = (1/2) * (sympy.diff(x_m, t)**2 + sympy.diff(y_m, t)**2)

# Kinetic Energy
T = W_cart + W_m

############
# Lagrange #
############

L = T - V

x_dot = sympy.diff(x, t)
θ_dot = sympy.diff(θ, t)

# Equations of Motion
eq_y = sympy.diff(sympy.diff(L, sympy.diff(y, t)), t) - sympy.diff(L, y)
eq_x = sympy.diff(sympy.diff(L, sympy.diff(x, t)), t) - sympy.diff(L, x) - F
eq_x_dot = sympy.diff(sympy.diff(L, sympy.diff(x_dot, t)),
                      t) - sympy.diff(L, x_dot)
eq_θ = sympy.diff(sympy.diff(L, sympy.diff(θ, t)), t) - sympy.diff(L, θ)
eq_θ_dot = sympy.diff(sympy.diff(L, sympy.diff(θ_dot, t)),
                      t) - sympy.diff(L, θ_dot)


A = sympy.Matrix([
    [sympy.diff(eq_y, y), sympy.diff(eq_x, y), sympy.diff(eq_x_dot, y),
     sympy.diff(eq_θ, y), sympy.diff(eq_θ_dot, y)],
    [sympy.diff(eq_y, x), sympy.diff(eq_x, x), sympy.diff(eq_x_dot, x),
     sympy.diff(eq_θ, x), sympy.diff(eq_θ_dot, x)],
    [sympy.diff(eq_y, x_dot), sympy.diff(eq_x, x_dot), sympy.diff(eq_x_dot, x_dot),
     sympy.diff(eq_θ, x_dot), sympy.diff(eq_θ_dot, x_dot)],
    [sympy.diff(eq_y, θ), sympy.diff(eq_x, θ), sympy.diff(eq_x_dot, θ),
     sympy.diff(eq_θ, θ), sympy.diff(eq_θ_dot, θ)],
    [sympy.diff(eq_y, θ_dot), sympy.diff(eq_x, θ_dot), sympy.diff(eq_x_dot, θ_dot),
     sympy.diff(eq_θ, θ_dot), sympy.diff(eq_θ_dot, θ_dot)]
])

linearize = [(sympy.sin(θ), 0), (sympy.cos(θ), 1), (θ.diff(t), 0),
             (θ.diff(t)**2, 0), (θ.diff(t, t), 0), (y.diff(t), 0), (y.diff(t, t), 0)]

A_lin = sympy.simplify(A.subs(linearize))

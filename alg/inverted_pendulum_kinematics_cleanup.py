from vpython import *
import control
import control.matlab
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
# import slycot
import sympy
import threading
import tqdm


def main():
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
    ω_ow = φ_ow.diff(t)

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

    # Potential Energy
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
    W_ow = (1/2) * m_ow * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2)
    W_iw = (1/2) * m_iw * (sympy.diff(x_iw, t)**2 + sympy.diff(y_iw, t)**2)
    W_m = (1/2) * m_m * (sympy.diff(x_m, t)**2 + sympy.diff(y_m, t)**2)

    # Dampening (e.g. heat dissipation)
    d_ow, d_iw, d_m, d_owr, d_iwr = sympy.symbols(
        'd_ow d_iw d_m d_owr d_iwr')

    W_d_ow = sympy.integrate(
        d_ow * (1/2) * (sympy.diff(x_ow, t)**2 + sympy.diff(y_ow, t)**2), t)
    W_d_iw = sympy.integrate(d_iw * (1/2) * sympy.diff(ρ, t)**2, t)
    W_d_m = sympy.integrate((d_m * (1/2) * sympy.diff(θ, t)**2), t)
    W_d_owr = sympy.integrate(d_owr * (1/2) * ω_ow**2, t)
    W_d_iwr = sympy.integrate(d_iwr * (1/2) * ω_iw**2, t)

    W_heat = W_d_ow + W_d_iw + W_d_m  # + W_d_owr + W_d_iwr
    # W_heat = 0

    # Kinetic Energy
    T = W_ow + W_iw + W_m  # + W_owr + W_iwr # + W_heat

    # TODO: Centrifugal forces seem missing

    ###############
    # State Space #
    ###############

    states = [θ, θ.diff(t), ρ, ρ.diff(t), φ, φ.diff(t)]

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

    eqs = [f_θ, f_dθ, f_ρ, f_dρ, f_φ, f_dφ]

    def createMatrix(eqs: list, states: list) -> sympy.Matrix:
        if (len(eqs) != len(states)):
            print("eqs and states must have the same size")
        A = sympy.zeros(len(eqs), len(eqs))
        for i, eq in enumerate(eqs, start=0):
            for j, state in enumerate(states, start=0):
                A[i, j] = sympy.diff(eq, state)
        return A

    # Create A Matrix
    A = createMatrix(eqs, states)

    def lineariceA(A_in, x_lineraice, x_delta):
        # Linearice
        linearice = [
            (sympy.sin(θ), sin(x_lineraice[0])),
            (sympy.cos(θ), cos(x_lineraice[0])),
            (sympy.sin(ρ), sin(x_lineraice[2])),
            (sympy.cos(ρ), cos(x_lineraice[2])),
            (θ.diff(t, t), x_delta[1]),
            (θ.diff(t)**2, x_lineraice[1]**2),
            (θ.diff(t), x_lineraice[1]),
            (θ, x_lineraice[0]),
            (ρ.diff(t, t), x_delta[3]),
            (ρ.diff(t)**2, x_lineraice[3]**2),
            (ρ.diff(t), x_lineraice[3]),
            (ρ, x_lineraice[2]),
            (φ.diff(t, t),  x_delta[5]),
            (φ.diff(t), x_lineraice[5]),
            (φ, x_lineraice[4]),
            (y.diff(t, t), 0),
            (y.diff(t), 0),
            (y, 0)
        ]

        return sympy.simplify(A_in.subs(linearice))

    A_lin_up = lineariceA(A, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    A_lin_down = lineariceA(A, [math.pi, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])

    masses = {
        g: 9.81,
        m_ow: 1.0,
        m_ow_rot: 0.5,
        m_iw: 0.3,
        m_iw_rot: 0.1,
        m_m: 15.0
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

    def applyConstants(A_):
        return A_.subs(masses).subs(lengths).subs(dampening)

    # states = [θ, θ.diff(t), ρ, ρ.diff(t), φ, φ.diff(t)]
    jacobian_up = np.float64(applyConstants(A_lin_up))
    jacobian_down = np.float64(applyConstants(A_lin_down))

    A_j_up = jacobian_up
    A_j_down = jacobian_down

    # TODO: Create proper movement matrix (see page 302)
    B = np.float64(sympy.Matrix([0, -1, 0, 0, 0, 1]))

    C = control.ctrb(A_j_up, B)
    rank = linalg.matrix_rank(C)

    print("Rank is " + str(rank))

    O = control.obsv(A_j_up, C)

    Q = np.float64(np.diag([1, 1, 1, 1, 1, 1]))

    R = np.float64(sympy.Matrix([1]))

    A_j_up.shape
    B.shape
    Q.shape
    R.shape

    # K, S, E = control.matlab.lqr(A_j, B, Q, R)
    # print(K)

    K_save = np.float64([[-1.35133831e+04, -5.98789910e+03, 8.32433588e-01,
                          8.19822691e-03, -2.23606797e+00, -8.34913488e-01]])

    K = K_save
    print(K)

    K.shape

    # Visualization
    # Initial State
    x_0 = np.float64([0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
    # Target State
    w_r = np.float64([math.pi, 0.0, 0.0, 0.0, 0.0, 0.0])

    dt = 0.01

    timeline = np.arange(0., 5., dt)

    def update_last_y(store_y):
        last_y = store_y

    last_y = x_0
    update_last_y(x_0)

    A_constant = applyConstants(A)

    def applyJ(y, t):
        # A_local = np.float64(lineariceA(A_constant, y, y - last_y))
        A_local = np.float64(lineariceA(A_constant, y, [0, 0, 0, 0, 0, 0]))
        # if (y[0] > - math.pi / 4 and y[0] < math.pi / 4):
        #     A_local = A_j_up
        # else:
        #     A_local = A_j_down

        # A_local = A_j_up
        # update_last_y(y)
        return A_j_up.dot(y) - (K * B).dot(y)
        # return A_local.dot(y)

    solution = integrate.odeint(applyJ, x_0, timeline)

    # x_ow, y_ow, x_iw, y_iw, x_m, y_m
    positions = sympy.zeros(6, len(timeline))
    energy = sympy.zeros(3, len(timeline))

    for tp in tqdm.tqdm(range(len(timeline))):
        results = {
            (θ.diff(t, t), 0),
            (θ.diff(t), solution[tp, 1]),
            (θ, solution[tp, 0]),
            (ρ.diff(t, t), 0),
            (ρ.diff(t), solution[tp, 3]),
            (ρ, solution[tp, 2]),
            (φ.diff(t, t), 0),
            (φ.diff(t), solution[tp, 5]),
            (φ, solution[tp, 4]),
            (y.diff(t, t), 0),
            (y.diff(t), 0),
            (y, 0)
        }
        positions[0, tp] = x_ow.subs(lengths).subs(results)
        positions[1, tp] = y_ow.subs(lengths).subs(results)
        positions[2, tp] = x_iw.subs(lengths).subs(results)
        positions[3, tp] = y_iw.subs(lengths).subs(results)
        positions[4, tp] = x_m.subs(lengths).subs(results)
        positions[5, tp] = y_m.subs(lengths).subs(results)
        energy[0, tp] = T.subs(lengths).subs(
            masses).subs(dampening).subs(results)
        energy[1, tp] = V.subs(lengths).subs(
            masses).subs(dampening).subs(results)
        energy[2, tp] = L.subs(lengths).subs(
            masses).subs(dampening).subs(results)

    fig, axs = plt.subplots(3)

    for row, state in enumerate(states):
        axs[0].plot(timeline, solution[:, row], label=str(state))

    axs[1].plot(timeline, positions[0, :].T, label='x_ow')
    # axs[1].plot(timeline, positions[1, :].T, label='y_ow')
    axs[1].plot(timeline, positions[2, :].T, label='x_iw')
    axs[1].plot(timeline, positions[3, :].T, label='y_iw')
    axs[1].plot(timeline, positions[4, :].T, label='x_m')
    axs[1].plot(timeline, positions[5, :].T, label='y_m')

    axs[2].plot(timeline, energy[0, :].T, label='T')
    axs[2].plot(timeline, energy[1, :].T, label='V')
    axs[2].plot(timeline, energy[2, :].T, label='L')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()

    plt.xlabel('t')
    # plt.axis([-0.01, 5, -0.75, 1.5])

    scene = canvas(align='left',
                   width=1830, height=1120,
                   center=vector(0, 0, 0), background=color.black)

    floor = box(pos=vector(0, -2.125, 0), length=8,
                height=0.25, width=4, color=color.gray(0.5))

    rod_m = cylinder(pos=vector(0, -1.0, -0.1),
                     axis=vector(0, 0, 0.6), radius=0.4, color=color.green)
    rod_iw = cylinder(pos=vector(0, -1.0, -0.1),
                      axis=vector(0, 0, 0.6), radius=r_1.subs(lengths), color=color.gray(0.2))
    rod_ow = cylinder(pos=vector(0, -1.0, 0),
                      axis=vector(0, 0, 0.4), radius=r_3.subs(lengths), color=color.gray(0.6))

    def render_task():
        while 1:
            for i in range(len(timeline)):
                rate(1 / dt)
                rod_ow.pos.x = positions[0, i]
                rod_ow.pos.y = positions[1, i] - 2
                rod_iw.pos.x = positions[2, i]
                rod_iw.pos.y = positions[3, i] - 2
                rod_m.pos.x = positions[4, i]
                rod_m.pos.y = positions[5, i] - 2

    render3d = threading.Thread(target=render_task)
    render3d.start()

    plt.show()


if __name__ == "__main__":
    main()

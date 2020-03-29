from burr import Burr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    B = Burr(2/3, 2)
    alph = 0.999
    Fu_min = 0.95
    q_true = B.var(alph)
    u_min = B.var(Fu_min)
    u_vals = np.linspace(u_min, q_true, 1000)
    q_approx = np.asarray([B.var_approx(u, alph) for u in u_vals])
    q_diff = q_true - q_approx
    q_bound = np.asarray([B.var_bound(u, alph) for u in u_vals])

    c_true = B.cvar(alph)
    c_approx = np.asarray([B.cvar_approx(u, alph) for u in u_vals])
    c_bound = np.asarray([B.cvar_bound(u, alph) for u in u_vals])
    c_diff = c_true - c_approx

    taus = np.asarray([B.tau(u) for u in u_vals])
    Fus = 1-taus

    plt.plot(Fus, q_diff)
    plt.plot(Fus, q_bound)
    plt.plot(Fus, c_diff)
    plt.plot(Fus, c_bound)
    plt.legend(labels=["q_diff", "q_bound", "c_diff", "c_bound"])
    plt.show()
    plt.clf()

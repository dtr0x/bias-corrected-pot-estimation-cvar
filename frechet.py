import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G, gammaincc as Ginc

def cdf(x, gamma):
    return np.exp(-x**(-gamma))

def rand(n, gamma):
    p = np.random.uniform(size=n)
    return var(p, gamma)

def var(alph, gamma):
    return (-np.log(alph))**(-1/gamma)

def cvar(alph, gamma):
    a = (gamma-1)/gamma
    x = -np.log(alph)
    return 1/(1-alph) * (G(a) - Ginc(a,x)*G(a))

def var_approx(u, alph, gamma):
    xi = 1/gamma
    sig = sigma(u,gamma)
    Fbar = tau(u,gamma)
    return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

def cvar_approx(u, alph, gamma):
    q = var_approx(u, alph, gamma)
    xi = 1/gamma
    sig = sigma(u,gamma)
    return q/(1-xi) + (sig-xi*u)/(1-xi)

def A(t, gamma):
    return -(1+gamma+t*gamma*np.log(1-1/t)) \
            /((t-1)*gamma*np.log(1-1/t)) - 1/gamma

def tau(u, gamma):
    return 1 - cdf(u, gamma)

def sigma(u, gamma):
    t = tau(u, gamma)
    return t*(-np.log(1-t))**(-1-1/gamma) / (1-t) / gamma

def s(u, alph, gamma):
    return tau(u, gamma)/(1-alph)

def I_ln_s(u, alph, gamma):
    t = s(u,alph,gamma)
    rho = -1
    xi = 1/gamma
    return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)

def var_bound(u, eta, alph, gamma):
    t = s(u,alph,gamma)
    rho = -1
    xi = 1/gamma
    L = sigma(u,gamma) * (eta+1) * t**eta * np.abs(A(1/tau(u,gamma),gamma))
    return L * I_ln_s(u, alph, gamma)

def cvar_bound(u, eta, alph, gamma):
    t = s(u,alph,gamma)
    rho = -1
    xi = 1/gamma
    L = sigma(u,gamma) * (eta+1) * t**eta * np.abs(A(1/tau(u,gamma),gamma))
    return L * (\
    t**(xi+rho)/rho/(xi+rho)/(1-eta-xi-rho) \
    - t**xi/rho/xi/(1-eta-xi) \
    + 1/xi/(xi+rho)/(1-eta) \
    )

def var_bounds(u, eta, alph, gamma):
    t = s(u,alph,gamma)
    rho = -1
    xi = 1/gamma
    sig = sigma(u,gamma)
    A_tau = A(1/tau(u,gamma),gamma)
    I_t = I_ln_s(u, alph, gamma)
    if A_tau >= 0:
        L = sig * (1-eta) * t**(-eta) * A_tau * I_t
        U = sig * (1+eta) * t**eta * A_tau * I_t
    elif A_tau < 0:
        L = sig * (1+eta) * t**eta * A_tau * I_t
        U = sig * (1-eta) * t**(-eta) * A_tau * I_t
    return L, U

def find_min_eta(Fu, alph, gamma):
    eta_vals = np.linspace(0, 1, 10000)[1:]
    u = var(Fu, gamma)
    q = var(alph, gamma)
    u_vals = np.linspace(u, q, 100)[:-1]
    for eta in eta_vals:
        q_approx = np.asarray([var_approx(u, alph, gamma) for u in u_vals])
        q_diff = q - q_approx
        q_bounds = np.asarray([var_bounds(u, eta, alph, gamma) for u in u_vals])
        L = q_bounds[:,0]
        U = q_bounds[:,1]
        intx = np.intersect1d(np.where(q_diff >= L), np.where(q_diff <= U))
        if  len(intx) > 0 and intx[-1] - intx[0] == len(u_vals) - 1:
            return eta
    return np.nan

if __name__ == '__main__':
    gamma = 1.5
    alph = 0.999
    Fu_min = 0.975
    eta = find_min_eta(Fu_min, alph, gamma)
    q_true = var(alph, gamma)
    u_init = var(Fu_min, gamma)
    u_vals = np.linspace(u_init, q_true, 1000)
    q_approx = np.asarray([var_approx(u, alph, gamma) for u in u_vals])
    q_bounds = np.asarray([var_bounds(u, eta, alph, gamma) for u in u_vals])
    L = q_bounds[:,0]
    U = q_bounds[:,1]
    q_diff = q_true - q_approx

    c_true = cvar(alph, gamma)
    c_approx = np.asarray([cvar_approx(u, alph, gamma) for u in u_vals])
    c_bound = np.asarray([cvar_bound(u, eta, alph, gamma) for u in u_vals])

    taus = np.asarray([tau(u, gamma) for u in u_vals])
    Fus = 1-taus

    #plt.plot(Fus, q_bound)
    #plt.plot(Fus, np.abs(q_true - q_approx))
    #plt.plot(Fus, c_bound)
    #plt.plot(Fus, np.abs(c_true - c_approx))
    #plt.legend(labels=["q_bound", "q_diff", "c_bound", "c_diff"])

    plt.plot(Fus, L)
    plt.plot(Fus, q_diff)
    plt.plot(Fus, U)
    plt.legend(labels=["L", "diff", "U"])
    print("Min eta: {:.4f}".format(eta))
    plt.show()
    plt.clf()

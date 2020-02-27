import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G, gammaincc as Ginc

def fr_cdf(x, gamma):
    return np.exp(-x**(-gamma))

def fr_rand(n, gamma):
    p = np.random.uniform(size=n)
    return fr_var(p, gamma)

def fr_var(alph, gamma):
    return (-np.log(alph))**(-1/gamma)

def fr_cvar(alph, gamma):
    a = (gamma-1)/gamma
    x = -np.log(alph)
    return 1/(1-alph) * (G(a) - Ginc(a,x)*G(a))

def fr_var_approx(u, alph, gamma):
    xi = 1/gamma
    sig = sigma(u,gamma)
    Fbar = tau(u,gamma)
    return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

def fr_cvar_approx(u, alph, gamma):
    q = fr_var_approx(u, alph, gamma)
    xi = 1/gamma
    sig = sigma(u,gamma)
    return q/(1-xi) + (sig-xi*u)/(1-xi)

def A(t, gamma):
    return -(1+gamma+t*gamma*np.log(1-1/t)) \
            /((t-1)*gamma*np.log(1-1/t)) - 1/gamma

def tau(u, gamma):
    return 1 - fr_cdf(u, gamma)

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

def frequency_func(cvars, cvar_true, eps):
    n = cvars.shape[0]
    abs_err = np.abs(cvars - cvar_true)
    return np.apply_along_axis(sum, 0, abs_err > eps)/n

if __name__ == '__main__':
    eta = 1e-9
    gamma = 2
    alph = 0.9995
    q_true = fr_var(alph, gamma)
    u_init = fr_var(0.9, gamma)
    u_vals = np.linspace(u_init, q_true, 1000)
    q_approx = np.asarray([fr_var_approx(u, alph, gamma) for u in u_vals])
    q_bound = np.asarray([var_bound(u, eta, alph, gamma) for u in u_vals])

    c_true = fr_cvar(alph, gamma)
    c_approx = np.asarray([fr_cvar_approx(u, alph, gamma) for u in u_vals])
    c_bound = np.asarray([cvar_bound(u, eta, alph, gamma) for u in u_vals])

    taus = np.asarray([tau(u, gamma) for u in u_vals])
    Fus = 1-taus

    plt.plot(Fus, q_bound)
    plt.plot(Fus, np.abs(q_true - q_approx))
    plt.plot(Fus, c_bound)
    plt.plot(Fus, np.abs(c_true - c_approx))
    plt.legend(labels=["q_bound", "q_diff", "c_bound", "c_diff"])

    plt.show()
    plt.clf()

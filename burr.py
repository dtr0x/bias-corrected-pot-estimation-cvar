import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from scipy.stats import burr12

def cdf(x, c, k):
    return 1 - (1 + x**c)**(-k)

def rand(n, c, k):
    return burr12.rvs(c, k, size=n)

def var(alph, c, k):
    return ((1-alph)**(-1/k) - 1)**(1/c)

def cvar(alph, c, k):
    Fv = 1-alph
    r = -1/c
    s = k-1/c
    t = 1-1/c+k
    return c*k/(c*k-1) * Fv**(-1/(c*k)) * hyp2f1(r, s, t, Fv**(1/k))

def var_approx(u, alph, c, k):
    xi = 1/c/k
    sig = sigma(u,c,k)
    Fbar = tau(u,c,k)
    return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

def cvar_approx(u, alph, c, k):
    q = var_approx(u, alph, c, k)
    xi = 1/c/k
    sig = sigma(u,c,k)
    return q/(1-xi) + (sig-xi*u)/(1-xi)

def A(t, c, k):
    return (1-c)/(c * k * (t**(1/k) - 1))

def tau(u, c, k):
    return 1 - cdf(u, c, k)

def sigma(u, c, k):
    t = tau(u,c,k)
    return 1/c/k * t**(-1/k) * (t**(-1/k) - 1)**(1/c - 1)

def s(u, alph, c, k):
    return tau(u, c, k)/(1-alph)

def I_ln_s(u, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)

def var_bound(u, eta, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    L = sigma(u,c,k) * (eta+1) * t**eta * np.abs(A(1/tau(u,c,k),c,k))
    return L * I_ln_s(u, alph, c, k)

def cvar_bound(u, eta, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    L = sigma(u,c,k) * (eta+1) * t**eta * np.abs(A(1/tau(u,c,k),c,k))
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
    c = 2
    k = 1
    alph = 0.9995
    q_true = var(alph, c, k)
    u_init = var(0.9, c, k)
    u_vals = np.linspace(u_init, q_true, 1000)
    q_approx = np.asarray([var_approx(u, alph, c, k) for u in u_vals])
    q_bound = np.asarray([var_bound(u, eta, alph, c, k) for u in u_vals])

    c_true = cvar(alph, c, k)
    c_approx = np.asarray([cvar_approx(u, alph, c, k) for u in u_vals])
    c_bound = np.asarray([cvar_bound(u, eta, alph, c, k) for u in u_vals])

    taus = np.asarray([tau(u, c, k) for u in u_vals])
    Fus = 1-taus

    plt.plot(Fus, q_bound)
    plt.plot(Fus, np.abs(q_true - q_approx))
    plt.plot(Fus, c_bound)
    plt.plot(Fus, np.abs(c_true - c_approx))
    plt.legend(labels=["q_bound", "q_diff", "c_bound", "c_diff"])

    plt.show()
    plt.clf()

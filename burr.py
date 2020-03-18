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

def var_bounds(u, eta, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    sig = sigma(u,c,k)
    A_tau = A(1/tau(u,c,k),c,k)
    I_t = I_ln_s(u, alph, c, k)
    if A_tau >= 0:
        L = sig * (1-eta) * t**(-eta) * A_tau * I_t
        U = sig * (1+eta) * t**eta * A_tau * I_t
    elif A_tau < 0:
        L = sig * (1+eta) * t**eta * A_tau * I_t
        U = sig * (1-eta) * t**(-eta) * A_tau * I_t
    return L, U

def var_bounds2(u, eta, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    sig = sigma(u,c,k)
    I_t = I_ln_s(u, alph, c, k)
    L = sig * (1-eta) * t**(-eta) * I_t
    U = sig * (1+eta) * t**eta * I_t
    return L, U

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

def cvar_bounds(u, eta, alph, c, k):
    t = s(u,alph,c,k)
    rho = -1/k
    xi = 1/c/k
    L = sigma(u,c,k) * (1-eta) * t**(-eta) * A(1/tau(u,c,k),c,k)
    U = sigma(u,c,k) * (1+eta) * t**eta * A(1/tau(u,c,k),c,k)
    L = L * (\
    t**(xi+rho)/rho/(xi+rho)/(1+eta-xi-rho) \
    - t**xi/rho/xi/(1+eta-xi) \
    + 1/xi/(xi+rho)/(1+eta) \
    )
    U = U * (\
    t**(xi+rho)/rho/(xi+rho)/(1-eta-xi-rho) \
    - t**xi/rho/xi/(1-eta-xi) \
    + 1/xi/(xi+rho)/(1-eta) \
    )
    if L > U:
        tmp = U
        U = L
        L = tmp
    return L, U

def find_min_eta(Fu, alph, c, k):
    eta_vals = np.linspace(0, 1, 10000)[1:]
    u = var(Fu, c, k)
    q = var(alph, c, k)
    u_vals = np.linspace(u, q, 1000)[:-1]
    for eta in eta_vals:
        q_approx = np.asarray([var_approx(u, alph, c, k) for u in u_vals])
        q_diff = q - q_approx
        q_bounds = np.asarray([var_bounds(u, eta, alph, c, k) for u in u_vals])
        L = q_bounds[:,0]
        U = q_bounds[:,1]
        intx = np.intersect1d(np.where(q_diff >= L), np.where(q_diff <= U))
        if len(intx) > 0 and intx[-1] - intx[0] == len(u_vals) - 1:
            return eta
    return np.nan



if __name__ == '__main__':
    c = 2
    k = 2
    alph = 0.999
    Fu_min = 0.975
    eta = find_min_eta(Fu_min, alph, c, k)
    q_true = var(alph, c, k)
    u_init = var(Fu_min, c, k)
    u_max = var(0.99999, c, k)
    u_vals = np.linspace(u_init, u_max, 1000)
    q_approx = np.asarray([var_approx(u, alph, c, k) for u in u_vals])
    q_diff = q_true - q_approx
    q_bounds = np.asarray([var_bounds(u, eta, alph, c, k) for u in u_vals])
    L_q = q_bounds[:,0]
    U_q = q_bounds[:,1]
    c_true = cvar(alph, c, k)
    c_approx = np.asarray([cvar_approx(u, alph, c, k) for u in u_vals])
    c_bounds = np.asarray([cvar_bounds(u, eta, alph, c, k) for u in u_vals])
    c_diff = c_true - c_approx
    L_c = c_bounds[:,0]
    U_c = c_bounds[:,1]


    taus = np.asarray([tau(u, c, k) for u in u_vals])
    Fus = 1-taus

    A_vals = np.asarray([A(1/t,c,k) for t in taus])

    plt.plot(Fus, L_q)
    plt.plot(Fus, q_diff)
    plt.plot(Fus, U_q)
    plt.legend(labels=["L", "diff", "U"])
    #plt.plot(Fus, q_bound)
    #plt.plot(Fus, np.abs(q_true - q_approx))
    #plt.plot(Fus, c_bound)
    #plt.plot(Fus, np.abs(c_true - c_approx))
    #plt.legend(labels=["q_bound", "q_diff", "c_bound", "c_diff"])
    print("Min eta: {:.4f}".format(eta))
    plt.show()
    plt.clf()

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from burr import *
from frechet import *
from eva import cvar_evt, cvar_sa
from asymp_var import asymp_var
from scipy.stats import norm

def cvar_iter_evt(x, alph, Fu, sampsizes):
    cvars = []
    for s in sampsizes:
        cvars.append(cvar_evt(x[:s], alph, Fu))
    return np.asarray(cvars)

def cvar_iter_sa(x, alph, sampsizes):
    cvars = []
    for s in sampsizes:
        cvars.append(cvar_sa(x[:s], alph))
    return np.asarray(cvars)

if __name__ == '__main__':
    np.random.seed(7)
    alph = 0.999
    c = 1.5
    k = 1
    gamma = 2
    Fu = 0.975
    s = 2000
    n = 40000
    step = 1000
    sampsizes = np.array([i for i in range(step, n+1, step)])

    Nus = np.asarray([(1-Fu)*n for n in sampsizes])

    burr_data = np.load("data/burr_data.npy")[0]
    frec_data = np.load("data/frec_data.npy")[0]

    burr_cvars_evt = np.load("data/burr_cvars_evt.npy")[1][1][0]
    frec_cvars_evt = np.load("data/frec_cvars_evt.npy")[1][1][0]

    B = Burr()
    F = Frechet()
    # thresholds at F(u)
    b_thresh = B.var(Fu, c, k)
    f_thresh = F.var(Fu, gamma)
    # B(u)
    b_cvar_bound = B.cvar_bounds(b_thresh, 0, alph, c, k)[0]
    f_cvar_bound = F.cvar_bounds(b_thresh, 0, alph, gamma)[0]
    # MDA shape parameter
    b_xi = 1/c/k
    f_xi = 1/gamma
    # Sigma values at F(u)
    b_sig = B.sigma(b_thresh, c, k)
    f_sig = F.sigma(f_thresh, gamma)
    # Asymptotic variances for the CVaR
    b_psi = asymp_var(b_xi, b_sig, Fu, alph)
    f_psi = asymp_var(f_xi, f_sig, Fu, alph)
    # Mean at each sample size, EVT
    b_cvar_evt_means = np.nanmean(burr_cvars_evt, axis=0)
    f_cvar_evt_means = np.nanmean(frec_cvars_evt, axis=0)

    b_ci_lower = b_cvar_evt_means - b_cvar_bound \
                - norm.ppf(0.975)*np.asarray([np.sqrt(b_psi/nu) for nu in Nus])
    b_ci_upper = b_cvar_evt_means - b_cvar_bound \
                + norm.ppf(0.975)*np.asarray([np.sqrt(b_psi/nu) for nu in Nus])

    f_ci_lower = f_cvar_evt_means - f_cvar_bound \
                - norm.ppf(0.975)*np.asarray([np.sqrt(f_psi/nu) for nu in Nus])
    f_ci_upper = f_cvar_evt_means - f_cvar_bound \
                + norm.ppf(0.975)*np.asarray([np.sqrt(f_psi/nu) for nu in Nus])

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(Nus, b_cvar_evt_means, linestyle='-', marker='.', color='r')
    axes[0].plot(Nus, b_ci_lower, linestyle=':', color='k')
    axes[0].plot(Nus, b_ci_upper, linestyle=':', color='k')
    #axes[0].set_ylim(60, 80)
    axes[1].plot(Nus, f_cvar_evt_means, linestyle='-', marker='.', color='b')
    axes[1].plot(Nus, f_ci_lower, linestyle=':', color='k')
    axes[1].plot(Nus, f_ci_upper, linestyle=':', color='k')
    #axes[1].set_ylim(60, 80)
    plt.show()
    fig.clf()

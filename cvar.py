import numpy as np
from scipy.stats import genpareto
from estimators import *
import warnings

def get_excesses(x, k):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    excesses = ord[-k:] - thresh
    return thresh, excesses

def gpd_fit(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi_mle, _, sig_mle = genpareto.fit(y, floc=0)
    return xi_mle, sig_mle

def get_params(x, k):
    u, y = get_excesses(x, k)
    xi_mle, sig_mle = gpd_fit(y)
    k_rho = sample_frac(len(x))
    rho = rho_est(x, k_rho)
    A = A_est(x, k, xi_mle, rho)
    return u, xi_mle, sig_mle, rho, A

def cvar_pot(x, alph, k, xi=None, sig=None, cutoff=0.9, debias=True):
    n = len(x)
    Fu  = 1 - k/n
    beta = (1-alph)/(1-Fu)
    if not (xi or sig):
        u, xi_mle, sig_mle, rho, A = get_params(x, k)
        if debias:
            xi, sig = debias_params(xi_mle, sig_mle, rho, A)
        else:
            xi, sig = xi_mle, sig_mle

    if xi > cutoff:
        return np.nan
    else:
        q = u + sig/xi * (beta**(-xi) - 1)
        cvar = (q + sig - xi*u)/(1-xi)

    if debias:
        approx_error = approx_error_est(xi, sig, rho, A, alph, n, k)
        cvar -= approx_error

    return cvar

def var_sa(x, alph):
    return np.sort(x)[int(np.floor(alph*len(x)))]

def cvar_sa(x, alph):
        q = var_sa(x, alph)
        y = x[x >= q]
        return np.mean(y)

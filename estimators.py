import numpy as np

'''Various estimators to compute auxilliary parameters related to bias correction.'''


def M_est(x, k, i):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    return np.mean(np.log(ord[(n-k):]/thresh)**i)


def A_est(x, k, xi, rho):
    a = M_est(x, k, 1)
    b = M_est(x, k, 2)
    return (xi+rho)/xi * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)


def T_est(x, k, tau=0):
    m1 = M_est(x, k, 1)
    m2 = M_est(x, k, 2)/2
    m3 = M_est(x, k, 3)/6
    if tau != 0:
        return (m1**tau - m2**(tau/2))/(m2**(tau/2) - m3**(tau/3))
    else:
        return (np.log(m1) - np.log(m2)/2)/(np.log(m2)/2 - np.log(m3)/3)


def rho_est(x, k, tau=0):
    T = T_est(x, k, tau)
    return 3*(T-1)/(T-3)


def mle_bias(xi, rho):
     return np.array([xi+1, -rho])/((1-rho)*(1+xi-rho))


def debias_params(xi_mle, sig_mle, rho, A):
    b = mle_bias(xi_mle, rho)
    xi = xi_mle - A*b[0]
    sig = sig_mle * (1-A*b[1])
    return xi, sig


def approx_error_est(xi, sig, rho, A, alph, n, k):
    s = k/n/(1-alph)
    if np.abs(xi+rho) != 0:
        x = s**xi/xi/(1-xi)
        y = s**(xi+rho)/(1-xi-rho) + rho/xi
        z = 1/(xi+rho)
        return sig*A*(x-z*y)/rho
    else:
        x = s**xi/xi/(1-xi)
        y = np.log(s)
        z = (xi-1)/xi
        return sig*A*(x-y+z)/rho

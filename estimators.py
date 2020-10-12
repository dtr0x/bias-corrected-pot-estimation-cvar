import numpy as np

def M_est(x, k, i):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    return np.mean(np.log(ord[(n-k):]/thresh)**i)

def A_est(x, k, xi, rho):
    a = M_est(x, k, 1)
    b = M_est(x, k, 2)
    return (xi+rho)/xi * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)

def T_est(x, k):
    a = np.log(M_est(x, k, 1))
    b = 1/2*np.log(M_est(x, k, 2)/2)
    c = 1/3*np.log(M_est(x, k, 3)/6)
    return (a-b)/(b-c)

def rho_est(x, k):
    t = T_est(x, k)
    return 3*(t-1)/(t-3)

def mle_bias_est(xi, rho):
     return np.array([xi+1, -rho])/((1-rho)*(1+xi-rho))

def debias_params(xi_mle, sig_mle, rho, A):
    b = mle_bias_est(xi_mle, rho)
    xi = xi_mle - A*b[0]
    sig = sig_mle * (1-A*b[1])
    return xi, sig

def approx_error_est(xi, sig, rho, A, alph, n, k):
    s = k/n/(1-alph)
    x1 = s**xi/xi/(1-xi)
    x2 = s**(xi+rho)/(xi+rho)/(1-xi-rho)
    x3 = rho/xi/(xi+rho)
    return sig * A * (x1 - x2 - x3) / rho

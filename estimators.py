import numpy as np
from scipy.special import gamma as G

def M_est(x, k, i):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    return np.mean(np.log(ord[(n-k):]/thresh)**i)

def M_conv(xi, i):
    return xi**i * G(i+1)

def A_est(x, k, xi):
    k_rho = sample_frac(len(x))
    rho = rho_est(x, k_rho)
    a = M_est(x, k, 1)
    b = M_est(x, k, 2)
    return (xi+rho)/xi * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)

def T_est(x, k):
    a = np.log(M_est(x, k, 1)/G(2))
    b = np.log(M_est(x, k, 2)/G(3))/2
    c = np.log(M_est(x, k, 3)/G(4))/3
    return (a-b)/(b-c)

def T_est2(x, k):
    a = np.log(M_est(x, k, 1))
    b = 1/2*np.log(M_est(x, k, 2)/2)
    c = 1/3*np.log(M_est(x, k, 3)/6)
    return (a-b)/(b-c)

def mu1(alph):
    return G(alph+1)

def mu2(alph,rho):
    return G(alph)*(1-(1-rho)**alph)/(rho*(1-rho)**alph)

def T_conv(rho):
    a = mu2(1,rho)/mu1(1)
    b = mu2(2,rho)/mu1(2)
    c = mu2(3,rho)/mu1(3)
    return (a-b)/(b-c)

def sample_frac(n):
    k = min(n-1, int(2*n/np.log(np.log(n))))
    return k

def rho_est(x, k):
    t = T_est(x, k)
    return 3*(t-1)/(t-3)

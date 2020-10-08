import numpy as np
# from scipy.special import gamma as G

def M_est(x, k, i):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    return np.mean(np.log(ord[(n-k):]/thresh)**i)

def M_conv(xi, i):
    return xi**i * G(i+1)

def A_est(x, k, xi, rho):
    a = M_est(x, k, 1)
    b = M_est(x, k, 2)
    return (xi+rho)/xi * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)

# def A_est(x, k, rho):
#     a = M_est(x, k, 1)
#     b = M_est(x, k, 2)
#     mom = a + 1 - 1/2 * (1-a**2/b)**(-1)
#     A = (mom+rho)/mom * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)
#     return A, mom

# def T_est(x, k):
#     a = np.log(M_est(x, k, 1)/G(2))
#     b = np.log(M_est(x, k, 2)/G(3))/2
#     c = np.log(M_est(x, k, 3)/G(4))/3
#     return (a-b)/(b-c)

def T_est(x, k):
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

def mle_bias_est(xi, rho):
     return np.array([xi+1, -rho])/((1-rho)*(1+xi-rho))

def debias_params(xi_mle, sig_mle, rho, A):
    b = mle_bias_est(xi_mle, rho)
    xi = xi_mle - A*b[0]
    sig = sig_mle * (1-A*b[1])
    return xi, sig

# def debias_params(x, k, xi_mle, sig_mle):
#     a = M_est(x, k, 1)
#     b = M_est(x, k, 2)
#     mom = a + 1 - 1/2 * (1-a**2/b)**(-1)
#     rho = rho_est(x, k)
#     A = (mom+rho)/mom * (1-rho)**2 * (b - 2*a**2) / (2*rho*a)
#     b = mle_bias_est(mom, rho)
#     xi = xi_mle - A*b[0]
#     sig = sig_mle/(1+A*b[1])
#     return xi, sig

def approx_error_est(xi, sig, rho, A, alph, n, k):
    s = k/n/(1-alph)
    x1 = s**xi/xi/(1-xi)
    x2 = s**(xi+rho)/(xi+rho)/(1-xi-rho)
    x3 = rho/xi/(xi+rho)
    return sig * A * (x1 - x2 - x3) / rho

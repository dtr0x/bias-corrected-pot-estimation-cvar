import numpy as np
from scipy.stats import genpareto

def avar_mle(xi, sig):
    var_sh = (1+xi)**2
    var_sc = 2*sig**2 * (1+xi)
    cov = -sig*(1+xi)
    return np.asarray([[var_sh, cov], [cov, var_sc]])

def Dsh(xi, sig, Fu, alph):
    b = (1-Fu)/(1-alph)
    x1 = 1/xi
    x2 = b**xi * np.log(b) / (1-xi)
    x3 = b**xi * (1-2*xi) / xi / (1-xi)**2
    return sig/xi * (x1 + x2 - x3)

def Dsc(xi, sig, Fu, alph):
    b = (1-Fu)/(1-alph)
    return (b**xi + xi - 1) / xi / (1-xi)

def asymp_var(xi, sig, Fu, alph):
    grad = np.asarray([Dsh(xi, sig, Fu, alph), Dsc(xi, sig, Fu, alph)])
    avmle = avar_mle(xi, sig)
    return np.matmul(np.matmul(grad, avmle), grad)

def bias(x, true):
    return np.mean(x, axis=0) - true

def mse(x, true):
    return np.mean((x-true)**2, axis=0)

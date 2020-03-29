import numpy as np

def avar_mle(xi, sig):
    var_sh = (1+xi)**2
    var_sc = sig**2 * (1 + (1+xi)**2)
    cov = -sig*(1+xi)
    return np.asarray([[var_sh, cov], [cov, var_sc]])

def Dsh(xi, sig, Fu, alph):
    b = (1-alph)/(1-Fu)
    x1 = 1/xi
    x2 = b**xi * np.log(b) / (1-xi)
    x3 = b**xi * (1-2*xi) / xi / (1-xi)**2
    return sig/xi**2 - sig*b**(-xi)*(1-2*xi+xi*(1-xi)*np.log(b)) \
            /xi**2/(1-xi)**2

def Dsc(xi, sig, Fu, alph):
    b = (1-alph)/(1-Fu)
    return (b**(-xi) + xi - 1) / xi / (1-xi)

def asymp_var(xi, sig, Fu, alph):
    grad = np.asarray([Dsh(xi, sig, Fu, alph), Dsc(xi, sig, Fu, alph)])
    avmle = avar_mle(xi, sig)
    return np.matmul(np.matmul(grad, avmle), grad)

def asymp_var_biased(xi, sig, xi_b, sig_b, Fu, alph):
    grad = np.asarray([Dsh(xi_b, sig_b, Fu, alph), Dsc(xi_b, sig_b, Fu, alph)])
    avmle = avar_mle(xi, sig)
    return np.matmul(np.matmul(grad, avmle), grad)

def bias(x, true):
    return np.nanmean(x, axis=0) - true

def mse(x, true):
    return np.nanmean((x-true)**2, axis=0)

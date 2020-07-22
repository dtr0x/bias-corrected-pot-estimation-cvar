import numpy as np

def avar_mle(xi):
    var_xi = (1+xi)**2
    var_sig = 1 + var_xi
    cov = -(1+xi)
    return np.asarray([[var_xi, cov], [cov, var_sig]])

def d_xi(xi, beta):
    return beta**xi * (2*xi + xi*(1-xi)*np.log(beta) - 1) / (xi*(1-xi))**2

def d_sig(xi, beta):
    return (beta**xi + xi - 1) / (xi*(1-xi))

def asymp_var(xi, beta):
    grad = np.asarray([d_xi(xi, beta), d_sig(xi, beta)])
    avmle = avar_mle(xi)
    return np.matmul(np.matmul(grad, avmle), grad)

import numpy as np
from scipy.stats import norm

'''Compute confidence intervals for the unbiased POT estimator'''

# asymptotic variance of maximum likelihood estimators
def avar_mle(xi):
    var_xi = (1+xi)**2
    var_sig = 1 + var_xi
    cov = -(1+xi)
    return np.asarray([[var_xi, cov], [cov, var_sig]])


# gradient term wrt xi
def d_xi(xi, beta):
    return beta**xi * (2*xi + xi*(1-xi)*np.log(beta) - 1) / (xi*(1-xi))**2


# gradient term wrt sigma
def d_sig(xi, beta):
    return (beta**xi + xi - 1) / (xi*(1-xi))


# asymptotic variance of (scaled) POT CVaR estimate
def V(xi, beta):
    grad = np.asarray([d_xi(xi, beta), d_sig(xi, beta)])
    avmle = avar_mle(xi)
    return np.matmul(np.matmul(grad, avmle), grad) + 1


# confidence intervals
def conf_int(upot, sig_est, var_est, k, delt=0.05):
    std_err = sig_est * np.sqrt(var_est/k)
    ci_lower = upot - std_err*norm.ppf(1-delt/2)
    ci_upper = upot + std_err*norm.ppf(1-delt/2)
    return ci_lower, ci_upper

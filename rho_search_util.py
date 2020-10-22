import numpy as np
import multiprocessing as mp
from estimators import rho_est

N_CPUs = mp.cpu_count()
POOL = mp.Pool(N_CPUs)

def theta(n, k):
    return np.around(np.log(k)/np.log(n), 2)

def mse(x, true):
    return np.nanmean((x-true)**2, axis=0)

def rho_mse(data, k, rho):
    result = [POOL.apply_async(rho_est, args=(x, k)) for x in data]
    r_est = np.array([r.get() for r in result])
    return np.mean(r_est), mse(r_est, rho)

def rho_search(data, rho, step=50):
    n = data.shape[1]
    k_init = n-1-int((n-1)/step)*step
    k = np.arange(k_init, n, step)
    r_est = []
    r_mse = []
    opt_i = 0
    for i in range(len(k)):
        r, m = rho_mse(data, k[i], rho)
        r_est.append(r)
        r_mse.append(m)
        if m == min(r_mse):
            opt_i = i

    return r_est[opt_i], r_mse[opt_i], k[opt_i]

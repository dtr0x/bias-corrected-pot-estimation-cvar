import numpy as np
from eva import *
from asymp_var import *
import multiprocessing as mp
from burr import *
import sys

if __name__ == '__main__':
    np.random.seed(7)
    #B = Burr(2/3, 2)
    B = Burr(1, 2)
    s = 2000
    n = 100000
    n_excesses = 2500
    xi = B.xi
    sig = B.aux_fun(n/n_excesses)
    parms_mle = np.array((xi, sig)) + B.mle_bias(n, n_excesses)
    data = B.rand((s, n))

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)

    result = [pool.apply_async(gpdFit,
                args=(x, n_excesses)) for x in data]
    params = []
    for r in result:
        params.append(r.get())
    params = np.asarray(params)
    params_avg = np.mean(params, axis=0)

    avar = avar_mle(xi, sig)
    m = mse(params, (xi, sig))
    b = bias(params, (xi, sig))
    crb = np.asarray([avar[0,0], avar[1,1]])/n_excesses
    eff = crb/m

    avar2 = avar_mle(parms_mle[0], parms_mle[1])
    m2 = mse(params, parms_mle)
    b2 = bias(params, parms_mle)
    crb2 = np.asarray([avar2[0,0], avar2[1,1]])/n_excesses
    eff2 = crb2/m2

    print("WITH TRUE VALS:")
    print("True (Xi, Sigma): {:.3f}, {:.3f}.".format(xi, sig))
    print("Mean (Xi, Sigma): {:.3f}, {:.3f}.".format(params_avg[0], params_avg[1]))
    print("Bias (Xi, Sigma): {:.3f}, {:.3f}.".format(b[0], b[1]))
    print("Efficiency (Xi, Sigma): {:.3f}, {:.3f}.".format(eff[0], eff[1]))

    print("WITH BIASED VALS:")
    print("True (Xi, Sigma): {:.3f}, {:.3f}.".format(parms_mle[0], parms_mle[1]))
    print("Mean (Xi, Sigma): {:.3f}, {:.3f}.".format(params_avg[0], params_avg[1]))
    print("Bias (Xi, Sigma): {:.3f}, {:.3f}.".format(b2[0], b2[1]))
    print("Biased Efficiency (Xi, Sigma): {:.3f}, {:.3f}.".format(eff2[0], eff2[1]))

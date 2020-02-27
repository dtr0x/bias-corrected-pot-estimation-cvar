import numpy as np
from eva import rgpd, gpdFit
from asymp_var import *
import multiprocessing as mp

if __name__ == '__main__':
    np.random.seed(7)
    xi = 0.25
    sig = 1
    s = 10000
    n = 1000
    data = rgpd((s, n), xi, sig)

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)

    result = [pool.apply_async(gpdFit,
                args=(x, 0)) for x in data]
    params = []
    for r in result:
        params.append(r.get())
    params = np.asarray(params)

    avar = avar_mle(xi, sig)
    b = bias(params, (xi, sig))
    crb = np.asarray([avar[0,0], avar[1,1]])/n
    eff = crb/mse(params, (xi, sig))

    print("Bias (Xi, Sigma): {:.3f}, {:.3f}.".format(b[0], b[1]))
    print("Efficiency (Xi, Sigma): {:.3f}, {:.3f}.".format(eff[0], eff[1]))

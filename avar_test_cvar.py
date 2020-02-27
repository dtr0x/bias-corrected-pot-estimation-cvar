import numpy as np
from eva import rgpd, cvar_gpd, cvar_evt
from asymp_var import *
import multiprocessing as mp

if __name__ == '__main__':
    np.random.seed(7)
    xi = 0.25
    sig = 1
    alph = 0.99
    Fu = 0
    cvar_true = cvar_gpd(alph, xi, sig)
    s = 10000
    n = 1000
    data = rgpd((s, n), xi, sig)

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)

    result = [pool.apply_async(cvar_evt,
                args=(x, alph, Fu)) for x in data]
    params = []
    for r in result:
        params.append(r.get())
    params = np.asarray(params)

    avar = asymp_var(xi, sig, Fu, alph)
    b = bias(params, cvar_true)
    crb = avar/n
    eff = crb/mse(params, cvar_true)

    print("Bias: {:.3f}".format(b))
    print("Efficiency: {:.3f}".format(eff))

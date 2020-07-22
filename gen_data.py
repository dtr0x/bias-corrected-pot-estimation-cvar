import numpy as np
import multiprocessing as mp
from cvar import cvar_pot
from asymp_var import asymp_var
from os.path import isfile

def sample_iter(x, alph, sampsizes, n_excesses, dist):
    cvar_pot_est = []
    xi_est = []
    sig_est = []
    approx_error = []
    avar = []
    for i in range(len(sampsizes)):
        n = sampsizes[i]
        k = n_excesses[i]
        t = n/k
        beta = k/(n*(1-alph))
        c, xi, sig = cvar_pot(x[:n], alph, k, dist)
        if (np.isnan(c)):
            cvar_pot_est.append(c)
            xi_est.append(c)
            sig_est.append(c)
            approx_error.append(c)
            avar.append(c)
        else:
            cvar_pot_est.append(c)
            xi_est.append(xi)
            sig_est.append(sig)
            approx_error.append(dist.cvar_approx_error(t, alph, xi, sig))
            avar.append(asymp_var(xi, beta)/k)

    return cvar_pot_est, xi_est, sig_est, approx_error, avar

def gen_data(dists, alph, n_samples, sampsizes, n_excesses):
    np.random.seed(7)
    n = sampsizes[-1]

    if isfile('data/calc.npz'):
        data = np.load('data/calc.npz')
        cvar_pot_est = data['cvar_pot_est']
        xi_est = data['xi_est']
        sig_est = data['sig_est']
        approx_error = data['approx_error']
        avar = data['avar']
    else:
        if isfile('data/samples.npz'):
            data = np.load('data/samples.npz')
            samples = data['samples']
        else:
            samples = []
            for d in dists:
                samples.append(d.rand((n_samples, n)))
            samples = np.asarray(samples)
            np.savez_compressed('data/samples.npz', samples=samples)

        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        cvar_pot_est = []
        xi_est = []
        sig_est = []
        approx_error = []
        avar = []
        for s, d in zip(samples, dists):
            result = [pool.apply_async(sample_iter,
                        args=(x, alph, sampsizes, n_excesses, d)) for x in s]
            c_d = []
            xi_d = []
            sig_d = []
            ae_d = []
            av_d = []
            for r in result:
                c, xi, si, ae, av = r.get()
                c_d.append(c)
                xi_d.append(xi)
                sig_d.append(si)
                ae_d.append(ae)
                av_d.append(av)
            cvar_pot_est.append(np.asarray(c_d))
            xi_est.append(np.asarray(xi_d))
            sig_est.append(np.asarray(sig_d))
            approx_error.append(np.asarray(ae_d))
            avar.append(np.asarray(av_d))

        cvar_pot_est=np.asarray(cvar_pot_est)
        xi_est=np.asarray(xi_est)
        sig_est=np.asarray(sig_est)
        approx_error=np.asarray(approx_error)
        avar=np.asarray(avar)
        np.savez_compressed('data/calc.npz', cvar_pot_est=cvar_pot_est, \
        xi_est=xi_est, sig_est=sig_est, approx_error=approx_error, avar=avar)

    return cvar_pot_est, xi_est, sig_est, approx_error, avar

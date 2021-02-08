import numpy as np
from frechet import Frechet
from burr import Burr
from half_t import HalfT
import multiprocessing as mp
from cvar_iter import cvar_iter


# generate samples from distributions
def gen_samples(dists, s, n):
    data = []
    for d in dists:
        data.append(d.rand((s, n)))
    return np.array(data)


# generate CVaR estimates from sample data
def get_cvars(dist_data, alph, sampsizes):
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    sa = [] # sample average estimates
    bpot = [] # biased POT estimates
    upot = [] # unbiased POT estimates
    #parameter estimates
    xi = []
    sigma = []
    rho = []
    k = []
    ae = []

    for d in dist_data:
        result = [pool.apply_async(cvar_iter, args=(x, alph, sampsizes)) for x in d]
        result = np.array([r.get() for r in result])
        sa.append(result[:,:,0])
        bpot.append(result[:,:,1])
        upot.append(result[:,:,2])
        xi.append(result[:,:,3])
        sigma.append(result[:,:,4])
        rho.append(result[:,:,5])
        k.append(result[:,:,6])
        ae.append(result[:,:,7])

    return np.array([sa, bpot, upot, xi, sigma, rho, k, ae])

if __name__ == '__main__':
    # CVaR level
    alph = 0.998

    xi = 2/3
    d = np.array([4, 3, 2.25, 0.75, 0.45])
    c = 1/(xi*d)
    burr_params = np.around([(i,j) for i,j in zip(c,d)], 2)
    fs_parms = np.linspace(1.5, 2.5, 5)
    dists = [Burr(*p) for p in burr_params]
    dists += [Frechet(p) for p in fs_parms]
    dists += [HalfT(p) for p in fs_parms]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(5000, 50000, 10).astype(int)
    n_max = sampsizes[-1]

    # number of independent runs
    s = 1000

    # generate data
    np.random.seed(0)
    data = gen_samples(dists, s, n_max)
    np.save('data/samples.npy', data)

    # get CVaR and parameter estimates
    result = get_cvars(data, alph, sampsizes)
    np.save('data/cvars.npy', result)

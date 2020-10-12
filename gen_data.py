from frechet import Frechet
from burr import Burr
import multiprocessing as mp
from cvar_multiprocess import *

# generate samples from distributions
def gen_samples(dists, s, n, seed=0):
    data = []
    for d in dists:
        np.random.seed(seed)
        data.append(d.rand((s, n)))
    return np.array(data)

# generate CVaR estimates from sample data
def get_cvars(dist_data, alph, sampsizes, k, k_rho):
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    cvars_upot = [] # unbiased POT estimates
    cvars_bpot = [] # biased POT estimates
    cvars_sa = [] # sample average estimates
    for d, kr in zip(dist_data, k_rho):
        result1 = [pool.apply_async(sample_iter_pot, args=(x, alph, sampsizes, k, True, kr)) for x in d]
        result2 = [pool.apply_async(sample_iter_pot, args=(x, alph, sampsizes, k, False)) for x in d]
        result3 = [pool.apply_async(sample_iter_sa, args=(x, alph, sampsizes)) for x in d]
        c_est1 = np.array([r.get() for r in result1])
        c_est2 = np.array([r.get() for r in result2])
        c_est3 = np.array([r.get() for r in result3])
        cvars_upot.append(c_est1)
        cvars_bpot.append(c_est2)
        cvars_sa.append(c_est3)

    return np.array([cvars_upot, cvars_bpot, cvars_sa])

if __name__ == '__main__':
    # Burr distributions
    xi = 2/3
    rhos = np.linspace(-0.25, -2, 8)
    d = -1/rhos
    c = 1/(xi*d)
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]

    # Frechet distributions
    gamma = np.linspace(1.25, 3, 8)
    frec_dists = [Frechet(p) for p in gamma]

    # sample sizes to test CVaR estimation
    N = np.array([10000, 20000, 30000, 40000, 50000])
    n_max = N[-1]

    # number of samples
    s = 1000

    # burr hyperparams
    k_b_rho = [np.ceil(N**0.85).astype(int),
                np.ceil(N**0.92).astype(int),
                N-1,
                np.ceil(N**0.97).astype(int),
                np.ceil(N**0.97).astype(int),
                np.ceil(N**0.97).astype(int),
                np.ceil(N**0.97).astype(int),
                np.ceil(N**0.98).astype(int)]

    # frechet hyperparams
    k_f_rho = [N-1]*len(frec_dists)

    k = np.ceil(N**0.8).astype(int)

    # generate data
    burr_data = gen_samples(burr_dists, s, n_max)
    frec_data = gen_samples(frec_dists, s, n_max)

    # CVaR level
    alph = 0.999

    burr_result = get_cvars(burr_data, alph, N, k, k_b_rho)
    frec_result = get_cvars(frec_data, alph, N, k, k_f_rho)

    np.save('data/burr_samples.npy', burr_data)
    np.save('data/frec_samples.npy', frec_data)
    np.save('data/burr_cvars.npy', burr_result)
    np.save('data/frec_cvars.npy', frec_result)

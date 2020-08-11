import numpy as np
import multiprocessing as mp
from gen_data import sample_iter_sa

if __name__ == '__main__':
    data = np.load('data/samples.npz')
    samples = data['samples']
    sampsizes = np.array([1000, 2000, 5000, 10000, 25000, 50000, 100000])
    alph = 0.999
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    cvars = []
    for s in samples:
        result = [pool.apply_async(sample_iter_sa,
            args=(x, alph, sampsizes)) for x in s]
        c = []
        for r in result:
            c.append(r.get())
        cvars.append(np.asarray(c))
    cvars = np.asarray(cvars)
    np.savez_compressed('data/cvar_sa.npz', cvars=cvars)

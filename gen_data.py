import numpy as np
import multiprocessing as mp
from eva import cvar_evt
from os.path import isfile

def cvar_iter(x, alph, sampsizes, n_excesses):
    cvars = []
    for i in range(len(sampsizes)):
        cvars.append(cvar_evt(x[:sampsizes[i]], alph, n_excesses[i]))
    return np.asarray(cvars)

def gen_data(burr_dists, frec_dists, alph, n_samples, sampsizes, n_excesses):
    np.random.seed(7)
    n = sampsizes[-1]

    # generate random Burr variates
    try:
        burr_data = np.load("data/burr_data.npy")
    except FileNotFoundError:
        burr_data = []
        for d in burr_dists:
            burr_data.append(d.rand((n_samples, n)))
        burr_data = np.asarray(burr_data)
        np.save("data/burr_data.npy", burr_data)

    # generate random Frechet variates
    try:
        frec_data = np.load("data/frec_data.npy")
    except FileNotFoundError:
        frec_data = []
        for d in frec_dists:
            frec_data.append(d.rand((n_samples, n)))
        frec_data = np.asarray(frec_data)
        np.save("data/frec_data.npy", frec_data)

    # EVT CVaR data for Burr distributions
    try:
        burr_cvars_evt = np.load("data/burr_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        burr_cvars_evt = []
        for data in burr_data:
            result = [pool.apply_async(cvar_iter,
                        args=(x, alph, sampsizes, n_excesses)) for x in data]
            cvars = []
            for r in result:
                cvars.append(r.get())
            burr_cvars_evt.append(np.asarray(cvars))
        burr_cvars_evt = np.asarray(burr_cvars_evt)
        np.save("data/burr_cvars_evt.npy", burr_cvars_evt)

    # EVT CVaR data for Frechet distributions
    try:
        frec_cvars_evt = np.load("data/frec_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        frec_cvars_evt = []
        for data in frec_data:
            result = [pool.apply_async(cvar_iter,
                        args=(x, alph, sampsizes, n_excesses)) for x in data]
            cvars = []
            for r in result:
                cvars.append(r.get())
            frec_cvars_evt.append(np.asarray(cvars))
        frec_cvars_evt = np.asarray(frec_cvars_evt)
        np.save("data/frec_cvars_evt.npy", frec_cvars_evt)

    return burr_data, frec_data, burr_cvars_evt, frec_cvars_evt

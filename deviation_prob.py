import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import burr, frechet
from eva import cvar_evt, cvar_sa
from asymp_var import asymp_var
from scipy.stats import norm

def cvar_iter_evt(x, alph, Fu, sampsizes):
    cvars = []
    for s in sampsizes:
        cvars.append(cvar_evt(x[:s], alph, Fu))
    return np.asarray(cvars)

def cvar_iter_sa(x, alph, sampsizes):
    cvars = []
    for s in sampsizes:
        cvars.append(cvar_sa(x[:s], alph))
    return np.asarray(cvars)

def frequency_func(cvars, cvar_true, eps):
    n = cvars.shape[0]
    abs_err = np.abs(cvars - cvar_true)
    return np.apply_along_axis(sum, 0, abs_err > eps)/n

if __name__ == '__main__':
    np.random.seed(7)
    alph = np.array([0.99, 0.999, 0.9999])
    c = np.array([2, 3])
    k = np.array([1, 2])
    gamma = np.array([2, 3, 4, 5])
    Fus = np.array([0.95, 0.975])
    eta = 1e-9
    err = 0.15
    s = 2000
    n = 40000
    step = 1000
    sampsizes = np.array([i for i in range(step, n+1, step)])

    # number of threshold excesses for each sample size
    Nus = np.asarray([[(1-fu)*n for n in sampsizes] for fu in Fus])

    # maximum domain of attraction Xi values
    burr_xi = np.asarray([[1/c_i/k_i for k_i in k] for c_i in c])
    frec_xi = np.asarray([1/g_i for g_i in gamma])

    # maximum domain of attraction Sigma values
    burr_sig = []
    frec_sig = []
    for fu in Fus:
        burr_sig.append(np.asarray([[burr.sigma(burr.var(fu, c_i, k_i), \
                        c_i, k_i) for k_i in k] for c_i in c]))
        frec_sig.append(np.asarray([frechet.sigma(frechet.var(fu, g_i), \
                        g_i) for g_i in gamma]))
    burr_sig = np.asarray(burr_sig)
    frec_sig = np.asarray(frec_sig)

    # asymptotic variances
    burr_psi = []
    frec_psi = []
    for i in range(len(Fus)):
        burr_psi_fu = []
        frec_psi_fu = []
        for a in alph:
            burr_psi_a = []
            frec_psi_a = []
            b_sh_fl = burr_xi.flatten()
            b_sc_fl = burr_sig[i].flatten()
            for j in range(len(b_sh_fl)):
                burr_psi_a.append(asymp_var(b_sh_fl[j], b_sc_fl[j], Fus[i], a))
                frec_psi_a.append(asymp_var(frec_xi[j], frec_sig[i][j], Fus[i], a))
            burr_psi_fu.append(np.asarray(burr_psi_a))
            frec_psi_fu.append(np.asarray(frec_psi_a))
        burr_psi.append(np.asarray(burr_psi_fu))
        frec_psi.append(np.asarray(frec_psi_fu))
    burr_psi = np.asarray(burr_psi)
    frec_psi = np.asarray(frec_psi)

    # true CVaR values
    burr_cvars_true = []
    frec_cvars_true = []
    for a in alph:
        burr_cvars_i = []
        frec_cvars_i = []
        for c_i in c:
            for k_i in k:
                burr_cvars_i.append(burr.cvar(a, c_i, k_i))
        burr_cvars_true.append(np.asarray(burr_cvars_i))
        for g_i in gamma:
            frec_cvars_i.append(frechet.cvar(a, g_i))
        frec_cvars_true.append(np.asarray(frec_cvars_i))
    burr_cvars_true = np.asarray(burr_cvars_true)
    frec_cvars_true = np.asarray(frec_cvars_true)

    # compute approximation error bounds
    burr_bounds = []
    frec_bounds = []
    for fu in Fus:
        burr_bounds_fu = []
        frec_bounds_fu = []
        for a in alph:
            burr_bounds_a = []
            frec_bounds_a = []
            for c_i in c:
                for k_i in k:
                    u = burr.var(fu, c_i, k_i)
                    burr_bounds_a.append(burr.cvar_bound(u, eta, a, c_i, k_i))
            burr_bounds_fu.append(np.asarray(burr_bounds_a))
            for g_i in gamma:
                u = frechet.var(fu, g_i)
                frec_bounds_a.append(frechet.cvar_bound(u, eta, a, g_i))
            frec_bounds_fu.append(np.asarray(frec_bounds_a))
        burr_bounds.append(np.asarray(burr_bounds_fu))
        frec_bounds.append(np.asarray(frec_bounds_fu))
    burr_bounds = np.asarray(burr_bounds)
    frec_bounds = np.asarray(frec_bounds)

    # theoretical deviation probabilities for each Fu, alpha, dist parameters
    z_burr = (np.array([burr_cvars_true]*2)*err - burr_bounds)/np.sqrt(burr_psi)
    z_frec = (np.array([frec_cvars_true]*2)*err - frec_bounds)/np.sqrt(frec_psi)
    burr_deviation_probs = []
    frec_deviation_probs = []
    for i in range(len(Fus)):
        std_norm_vals_burr = np.asarray([np.sqrt(nu)*z_burr[i] for nu in Nus[i]])
        std_norm_vals_frec = np.asarray([np.sqrt(nu)*z_frec[i] for nu in Nus[i]])
        burr_deviation_probs.append(2*(1-norm.cdf(std_norm_vals_burr)))
        frec_deviation_probs.append(2*(1-norm.cdf(std_norm_vals_frec)))
    burr_deviation_probs = np.asarray(burr_deviation_probs)
    frec_deviation_probs = np.asarray(frec_deviation_probs)

    # generate random Burr variates
    try:
        burr_data = np.load("data/burr_data.npy")
    except FileNotFoundError:
        burr_data = []
        for c_i in c:
            for k_i in k:
                burr_data.append(burr.rand((s, n), c_i, k_i))
        burr_data = np.asarray(burr_data)
        np.save("data/burr_data.npy", burr_data)

    # generate random Frechet variates
    try:
        frec_data = np.load("data/frec_data.npy")
    except FileNotFoundError:
        frec_data = []
        for g_i in gamma:
            frec_data.append(frechet.rand((s, n), g_i))
        frec_data = np.asarray(frec_data)
        np.save("data/frec_data.npy", frec_data)

    # SA CVaR data for Burr distributions
    try:
        burr_cvars_sa = np.load("data/burr_cvars_sa.npy")
    except FileNotFoundError:
        burr_cvars_sa = []
        for a in alph:
            cvars_sa_alph = []
            for data in burr_data:
                c_alph = lambda z: cvar_iter_sa(z, a, sampsizes)
                cvars_sa_alph.append(np.apply_along_axis(c_alph, 1, data))
            burr_cvars_sa.append(np.asarray(cvars_sa_alph))
        burr_cvars_sa = np.asarray(burr_cvars_sa)
        np.save("data/burr_cvars_sa.npy", burr_cvars_sa)

    # SA CVaR data for Frechet distributions
    try:
        frec_cvars_sa = np.load("data/frec_cvars_sa.npy")
    except FileNotFoundError:
        frec_cvars_sa = []
        for a in alph:
            cvars_sa_alph = []
            for data in frec_data:
                c_alph = lambda z: cvar_iter_sa(z, a, sampsizes)
                cvars_sa_alph.append(np.apply_along_axis(c_alph, 1, data))
            frec_cvars_sa.append(np.asarray(cvars_sa_alph))
        frec_cvars_sa = np.asarray(frec_cvars_sa)
        np.save("data/frec_cvars_sa.npy", frec_cvars_sa)

    # EVT CVaR data for Burr distributions
    try:
        burr_cvars_evt = np.load("data/burr_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        burr_cvars_evt = []
        i = 1
        t_total = 0
        for fu in Fus:
            cvars_evt_fu = []
            for a in alph:
                cvars_evt_alph = []
                for data in burr_data:
                    start = time.time()
                    result = [pool.apply_async(cvar_iter_evt,
                                args=(x, a, fu, sampsizes)) for x in data]
                    cvars = []
                    for r in result:
                        cvars.append(r.get())
                    cvars_evt_alph.append(np.asarray(cvars))
                    end = time.time()
                    t = (end - start)/60
                    print("Finished Burr data slice {} in {:.2f} minutes."\
                            .format(i, t))
                    i += 1
                    t_total += t
                cvars_evt_fu.append(np.asarray(cvars_evt_alph))
            burr_cvars_evt.append(np.asarray(cvars_evt_fu))
        burr_cvars_evt = np.asarray(burr_cvars_evt)
        np.save("burr_cvars_evt.npy", burr_cvars_evt)
        print("Finished calculating EVT CVaRs for Burr distributions \
                in {:.2f} minutes.".format(t_total))

    # EVT CVaR data for Frechet distributions
    try:
        frec_cvars_evt = np.load("data/frec_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        frec_cvars_evt = []
        i = 1
        t_total = 0
        for fu in Fus:
            cvars_evt_fu = []
            for a in alph:
                cvars_evt_alph = []
                for data in frec_data:
                    start = time.time()
                    result = [pool.apply_async(cvar_iter_evt,
                                args=(x, a, fu, sampsizes)) for x in data]
                    cvars = []
                    for r in result:
                        cvars.append(r.get())
                    cvars_evt_alph.append(np.asarray(cvars))
                    end = time.time()
                    t = (end - start)/60
                    print("Finished Frechet data slice {} in {:.2f} minutes."\
                            .format(i, t))
                    i += 1
                    t_total += t
                cvars_evt_fu.append(np.asarray(cvars_evt_alph))
            frec_cvars_evt.append(np.asarray(cvars_evt_fu))
        frec_cvars_evt = np.asarray(frec_cvars_evt)
        np.save("data/frec_cvars_evt.npy", frec_cvars_evt)
        print("Finished calculating EVT CVaRs for Frechet distributions \
                in {:.2f} minutes.".format(t_total))

    burr_parms = []
    for c_i in c:
        for k_i in k:
            burr_parms.append((c_i, k_i))
    # plot the deviation probability (theoretical bound, EVT, SA)
    for i in range(len(Fus)):
        for j in range(len(alph)):
            for m in range(len(burr_parms)):
                # Burr plots
                b_cv_e = burr_cvars_evt[i, j, m]
                b_cv_s = burr_cvars_sa[j, m]
                b_cv_t = burr_cvars_true[j, m]
                b_eps = err*b_cv_t
                b_prob_evt = frequency_func(b_cv_e, b_cv_t, b_eps)
                b_prob_sa = frequency_func(b_cv_s, b_cv_t, b_eps)
                b_prob_bound = burr_deviation_probs[i, :, j, m]
                plt.plot(sampsizes, b_prob_evt, linestyle='-', marker='.', color='r')
                plt.plot(sampsizes, b_prob_sa, linestyle=':', marker='.', fillstyle='none', color='b')
                plt.plot(sampsizes, b_prob_bound, color='k')
                plt.xlabel("sample size")
                plt.ylabel("relative frequency")
                plt.legend(labels=["EVT CVaR", "SA CVaR", "P"])
                plt.title("Burr{}, alpha={}, F(u)={}".format(\
                    burr_parms[m], alph[j], Fus[i]))
                plt.savefig("plots/burr/{}_{}_{}_{}.png".format(\
                    burr_parms[m][0], burr_parms[m][1], alph[j], Fus[i]),\
                    bbox_inches="tight")
                plt.clf()
                # Frechet plots
                f_cv_e = frec_cvars_evt[i, j, m]
                f_cv_s = frec_cvars_sa[j, m]
                f_cv_t = frec_cvars_true[j, m]
                f_eps = err*f_cv_t
                f_prob_evt = frequency_func(f_cv_e, f_cv_t, f_eps)
                f_prob_sa = frequency_func(f_cv_s, f_cv_t, f_eps)
                f_prob_bound = frec_deviation_probs[i, :, j, m]
                plt.plot(sampsizes, f_prob_evt, linestyle='-', marker='.', color='r')
                plt.plot(sampsizes, f_prob_sa, linestyle=':', marker='.', fillstyle='none', color='b')
                plt.plot(sampsizes, f_prob_bound, color='k')
                plt.xlabel("sample size")
                plt.ylabel("relative frequency")
                plt.legend(labels=["EVT CVaR", "SA CVaR", "P"])
                plt.title("Frechet({}), alpha={}, F(u)={}".format(\
                    gamma[m], alph[j], Fus[i]))
                plt.savefig("plots/frechet/{}_{}_{}.png".format(\
                    gamma[m], alph[j], Fus[i]),\
                    bbox_inches="tight")
                plt.clf()

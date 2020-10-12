from param_search_util import *
from burr import Burr

if __name__ == '__main__':
    s = 100
    n = 25000

    xi = 2/3
    rhos = np.linspace(-0.25, -2, 8)
    d = -1/rhos
    c = 1/(xi*d)
    params = [(i,j) for i,j in zip(c,d)]
    dists = [Burr(*p) for p in params]

    T = []
    k_vals = []
    for D in dists:
        np.random.seed(0)
        data = D.rand((s, n))
        r = D.rho
        r_est, r_mse, k_rho = rho_search(data, r)
        print("Burr({:.2f}, {:.2f}): ".format(D.c, D.d))
        print(r_est, r_mse, k_rho)
        print('')
        T.append(theta(n, k_rho))
        k_vals.append(k_rho)

    np.save('k_rho_burr.npy', np.array(k_vals))

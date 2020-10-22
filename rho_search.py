from rho_search_util import *
from burr import Burr

if __name__ == '__main__':
    s = 100
    n = 25000

    # Burr distributions
    xi = 2/3
    rho = -np.array([0.25, 0.4, 0.75, 1.5, 2])
    d = -1/rho
    c = 1/(xi*d)
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]

    T = []
    for D in burr_dists:
        np.random.seed(0)
        data = D.rand((s, n))
        r = D.rho
        r_est, r_mse, k_rho = rho_search(data, r)
        print("Burr({:.2f}, {:.2f}): ".format(D.c, D.d))
        print(r_est, r_mse, k_rho)
        print('')
        T.append(theta(n, k_rho))

    np.save('rho_samp_theta.npy', np.array(T))
